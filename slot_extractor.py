"""Config-driven slot extractor with staged matching and detailed attempt logs.

API: SlotExtractor(config, base_dir).extract_slots(user_utterance)
Pattern candidates are canonicalized exact -> alias -> Levenshtein -> DistilBERT.
Only if no pattern/attribute match succeeds are individual tokens scanned against
ontology slots. Pattern matching can skip configured leading determiners such as
'a', 'an' and 'the' before the slot value.
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any

WORD_RE = re.compile(r"[\w]+(?:['’][\w]+)*", re.UNICODE)
PATTERN_TOKEN_RE = re.compile(r"\{[\w ]+\}|[\w]+(?:['’][\w]+)*", re.UNICODE)
MATCH_STAGES = ("exact", "ontology_alias", "levenshtein", "distilbert")


def _norm(value: Any) -> str:
    return " ".join(str(value or "").casefold().replace("’", "'").split())


def _words(text: str) -> list[str]:
    return [m.group().casefold().replace("’", "'") for m in WORD_RE.finditer(text)]


def _load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def _similarity(a: str, b: str) -> float:
    a, b = _norm(a), _norm(b)
    return 1.0 - _levenshtein(a, b) / max(len(a), len(b), 1)


def _match_pattern(pattern: str, tokens: list[str], max_value_tokens: int,
                   leading_stopwords: set[str], max_leading_stopwords: int):
    """Yield slot-value spans, optionally skipping configured words after the prefix."""
    parts = [m.group().casefold().replace("’", "'") for m in PATTERN_TOKEN_RE.finditer(pattern)]
    placeholders = [i for i, part in enumerate(parts) if part.startswith("{") and part.endswith("}")]
    if len(placeholders) != 1:
        return
    p = placeholders[0]
    prefix, suffix = parts[:p], parts[p + 1:]
    for start in range(len(tokens) + 1):
        if tokens[start:start + len(prefix)] != prefix:
            continue
        prefix_end = start + len(prefix)
        for skipped in range(max_leading_stopwords + 1):
            if skipped and (prefix_end + skipped > len(tokens)
                            or any(token not in leading_stopwords
                                   for token in tokens[prefix_end:prefix_end + skipped])):
                break
            value_start = prefix_end + skipped
            for length in range(1, max_value_tokens + 1):
                end = value_start + length
                suffix_end = end + len(suffix)
                if end <= len(tokens) and suffix_end <= len(tokens):
                    if tokens[end:suffix_end] == suffix:
                        yield value_start, end


class SlotExtractor:
    """Instantiate with loaded config, then call extract_slots(user_utterance)."""
    def __init__(self, config: dict[str, Any], base_dir: Path | None = None):
        self.config = config
        self.base_dir = (base_dir or Path.cwd()).resolve()
        self.files = config["files"]
        self.matching = config["matching"]
        self.slots_config = config["slots"]
        self.attr_config = self.slots_config["attr"]
        self.interaction = config["interaction"]

        def resource_path(name: str) -> Path:
            path = Path(self.files[name])
            return path if path.is_absolute() else self.base_dir / path

        self.patterns = _load_json(resource_path("patterns_path"))
        self.aliases = _load_json(resource_path("aliases_path"))
        self.ontology = _load_json(resource_path("ontology_path"))
        self.slot_name_map = {_norm(k): v for k, v in self.slots_config["slot_name_map"].items()}
        self.allowed_slots = list(self.slots_config["allowed_slots"])
        self.values_by_slot: dict[str, set[str]] = {s: set() for s in self.allowed_slots}
        for raw_slot, values in self.ontology.items():
            slot = self._normalize_slot(raw_slot)
            if slot in self.values_by_slot:
                self.values_by_slot[slot].update(_norm(v) for v in values if _norm(v))
        self.values_by_slot.setdefault("attr", set()).update(
            _norm(v) for v in self.slots_config["attr_values"]
        )
        self.log_path = Path(self.interaction["log_path"])
        if not self.log_path.is_absolute():
            self.log_path = self.base_dir / self.log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._semantic = None
        self._errors: list[str] = []

    def _normalize_slot(self, slot: Any) -> str:
        key = _norm(slot)
        return self.slot_name_map.get(key, key)

    @staticmethod
    def _normalize_attr_text(text: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()

    def _attribute_candidates(self, utterance: str):
        token_matches = list(WORD_RE.finditer(utterance))
        found = []
        for canonical, aliases in self.attr_config["aliases"].items():
            for alias in aliases:
                normalized = self._normalize_attr_text(alias)
                if not normalized:
                    continue
                phrase = r"[\W_]+".join(re.escape(word) for word in normalized.split())
                regex = re.compile(rf"(?<!\w){phrase}(?!\w)", re.IGNORECASE)
                for match in regex.finditer(utterance):
                    ids = [i for i, token in enumerate(token_matches)
                           if token.start() < match.end() and token.end() > match.start()]
                    if ids:
                        found.append({"slot": "attr", "value": canonical, "surface": match.group(),
                                      "start": ids[0], "end": ids[-1] + 1, "pattern": None,
                                      "source": "attribute_alias", "method": "explicit_attribute_alias",
                                      "similarity": 1.0, "char_start": match.start(),
                                      "allow_special": False})
        found.sort(key=lambda x: (x["char_start"], -(x["end"] - x["start"])))
        unique, seen = [], set()
        for item in found:
            if item["value"] not in seen:
                seen.add(item["value"])
                unique.append(item)
        return unique if self.attr_config.get("multi_value", False) else unique[:1]

    def _alias_forms(self, canonical: str, slot: str) -> list[str]:
        entry = self.aliases.get(canonical, []) if isinstance(self.aliases, dict) else []
        if isinstance(entry, list):
            forms = [_norm(x) for x in entry]
        elif isinstance(entry, dict):
            forms = [_norm(x) for x in entry.get(slot, [])]
        else:
            forms = []
        return list(dict.fromkeys(x for x in forms if x))

    def _forms_for_slot(self, slot: str, allow_special: bool) -> list[tuple[str, str]]:
        forms = []
        for canonical in sorted(self.values_by_slot.get(slot, set())):
            forms.append((canonical, canonical))
            if self.matching["use_ontology_alias"]:
                forms.extend((alias, canonical) for alias in self._alias_forms(canonical, slot))
        if allow_special and self.matching["use_ontology_alias"]:
            for canonical, aliases in self.slots_config.get("special_aliases", {}).items():
                forms.extend((_norm(alias), _norm(canonical)) for alias in aliases)
        return list(dict.fromkeys((surface, canonical) for surface, canonical in forms if surface))

    def _semantic_runtime(self):
        if self._semantic is None:
            try:
                import torch
                from transformers import AutoModel, AutoTokenizer
            except ImportError as exc:
                raise RuntimeError("Semantic matching requires: pip install torch transformers") from exc
            name = self.matching["semantic_model_name"]
            tokenizer = AutoTokenizer.from_pretrained(name)
            model = AutoModel.from_pretrained(name)
            model.eval()
            self._semantic = (torch, tokenizer, model, {})
        return self._semantic

    def _semantic_match(self, surface: str, forms: list[tuple[str, str]]):
        if not forms:
            return None
        torch, tokenizer, model, cache = self._semantic_runtime()
        texts = list(dict.fromkeys([_norm(surface)] + [form for form, _ in forms]))
        missing = [text for text in texts if text not in cache]
        if missing:
            encoded = tokenizer(missing, padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                hidden = model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                vectors = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            for text, vector in zip(missing, vectors):
                cache[text] = vector.detach()
        query = cache[_norm(surface)].unsqueeze(0)
        candidates = torch.stack([cache[form] for form, _ in forms])
        scores = torch.nn.functional.cosine_similarity(query, candidates, dim=1)
        best = int(torch.argmax(scores).item())
        form, canonical = forms[best]
        return canonical, float(scores[best].item()), form

    def _match_stage(self, surface: str, slot: str, stage: str, allow_special: bool):
        surface = _norm(surface)
        forms = self._forms_for_slot(slot, allow_special)
        canonicals = self.values_by_slot.get(slot, set())

        def attempt(score, candidate=None, compared_form=None, threshold=None,
                    enabled=True, accepted=False, error=None):
            return {"stage": stage, "slot": slot, "surface": surface,
                    "candidate_canonical": candidate, "compared_form": compared_form,
                    "score": score, "threshold": threshold, "enabled": enabled,
                    "accepted": accepted, "error": error}

        if stage == "exact":
            candidate = next((canonical for _, canonical in forms
                              if surface == canonical and canonical in canonicals), None)
            result = (candidate, "exact", 1.0) if candidate else None
            return result, attempt(1.0 if candidate else 0.0, candidate, candidate,
                                  threshold=1.0, accepted=bool(candidate))

        if stage == "ontology_alias":
            if not self.matching["use_ontology_alias"]:
                return None, attempt(None, enabled=False, error="disabled in config")
            found = next(((canonical, form) for form, canonical in forms
                          if surface == form and surface != canonical), None)
            result = (found[0], "ontology_alias", 1.0) if found else None
            return result, attempt(1.0 if found else 0.0,
                                  found[0] if found else None,
                                  found[1] if found else None,
                                  threshold=1.0, accepted=bool(found))

        if stage == "levenshtein":
            enabled = bool(self.matching["use_levenshtein"])
            threshold = float(self.matching["levenshtein_threshold"])
            minimum = int(self.matching["min_fuzzy_chars"])
            if not enabled:
                return None, attempt(None, threshold=threshold, enabled=False, error="disabled in config")
            if len(surface.replace(" ", "")) < minimum:
                return None, attempt(0.0, threshold=threshold, error=f"shorter than min_fuzzy_chars={minimum}")
            best = None
            for form, canonical in forms:
                if len(form.replace(" ", "")) < minimum:
                    continue
                score = _similarity(surface, form)
                if best is None or score > best[2]:
                    best = (canonical, form, score)
            if best is None:
                return None, attempt(0.0, threshold=threshold, error="no eligible ontology forms")
            accepted = best[2] >= threshold
            result = (best[0], "levenshtein", best[2]) if accepted else None
            return result, attempt(best[2], best[0], best[1], threshold, accepted=accepted)

        if stage == "distilbert":
            enabled = bool(self.matching["use_semantic"])
            threshold = float(self.matching["semantic_threshold"])
            if not enabled:
                return None, attempt(None, threshold=threshold, enabled=False, error="disabled in config")
            try:
                best = self._semantic_match(surface, forms)
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                self._errors.append(f"DistilBERT error for {surface!r}: {message}")
                return None, attempt(None, threshold=threshold, error=message)
            if best is None:
                return None, attempt(0.0, threshold=threshold, error="no eligible ontology forms")
            canonical, score, form = best
            accepted = score >= threshold
            result = (canonical, "distilbert", score) if accepted else None
            return result, attempt(score, canonical, form, threshold, accepted=accepted)

        return None, attempt(None, error="unknown stage")

    def _pattern_candidates(self, utterance: str, tokens: list[str]):
        candidates = []
        max_tokens = int(self.matching["max_pattern_value_tokens"])
        leading = {_norm(x) for x in self.matching.get("pattern_leading_stopwords", [])}
        max_leading = int(self.matching.get("max_leading_stopwords", 0))
        for raw_slot, patterns in self.patterns.items():
            slot = self._normalize_slot(raw_slot)
            if slot not in self.allowed_slots or not isinstance(patterns, list):
                continue
            for pattern in patterns:
                for start, end in _match_pattern(str(pattern), tokens, max_tokens, leading, max_leading):
                    candidates.append({"slot": slot, "surface": " ".join(tokens[start:end]),
                                       "start": start, "end": end, "pattern": str(pattern),
                                       "source": "pattern", "allow_special": True})
        candidates.extend(self._attribute_candidates(utterance))
        unique = {}
        for item in candidates:
            key = (item["slot"], item["start"], item["end"], item.get("value"))
            if key not in unique or len(str(item.get("pattern") or "")) > len(str(unique[key].get("pattern") or "")):
                unique[key] = item
        return list(unique.values())

    def _apply_stages(self, candidates, matches, occupied, attempts_log):
        unresolved = list(candidates)
        for stage in MATCH_STAGES:
            next_unresolved = []
            for item in unresolved:
                if any(i in occupied for i in range(item["start"], item["end"])):
                    continue
                if item["source"] == "attribute_alias":
                    result = (item["value"], item["method"], item["similarity"]) if stage == "exact" else None
                    attempt = {"stage": "explicit_attribute_alias" if stage == "exact" else stage,
                               "slot": item["slot"], "surface": item["surface"],
                               "candidate_canonical": item["value"], "compared_form": item["surface"],
                               "score": 1.0 if stage == "exact" else None,
                               "threshold": None, "enabled": stage == "exact",
                               "accepted": stage == "exact",
                               "error": None if stage == "exact" else "not needed: explicit alias matched"}
                else:
                    result, attempt = self._match_stage(item["surface"], item["slot"], stage,
                                                        item.get("allow_special", False))
                attempt.update({"source": item["source"], "pattern": item.get("pattern"),
                                "token_start": item["start"], "token_end": item["end"]})
                item.setdefault("attempts", []).append(attempt)
                attempts_log.append(attempt)
                if result:
                    canonical, method, score = result
                    matches.append({**item, "value": canonical, "method": method,
                                    "similarity": float(score)})
                    occupied.update(range(item["start"], item["end"]))
                else:
                    next_unresolved.append(item)
            unresolved = next_unresolved
            if not unresolved:
                break
        return unresolved

    def _token_fallback(self, tokens: list[str], matches, occupied, attempts_log):
        for stage in MATCH_STAGES:
            found = []
            for index, token in enumerate(tokens):
                if index in occupied:
                    continue
                for slot in self.allowed_slots:
                    result, attempt = self._match_stage(token, slot, stage, allow_special=False)
                    attempt.update({"source": "ontology_token_scan", "pattern": None,
                                    "token_start": index, "token_end": index + 1})
                    attempts_log.append(attempt)
                    if result:
                        canonical, method, score = result
                        found.append({"slot": slot, "value": canonical, "surface": token,
                                      "start": index, "end": index + 1, "pattern": None,
                                      "source": "ontology_token_scan", "method": method,
                                      "similarity": float(score), "attempt": attempt})
                        break
            found.sort(key=lambda item: (item["start"], self.allowed_slots.index(item["slot"])))
            for item in found:
                if item["start"] not in occupied:
                    matches.append(item)
                    occupied.add(item["start"])

    def extract_slots(self, user_utterance: str) -> str:
        self._errors = []
        tokens = _words(user_utterance)
        matches, attempts_log, occupied = [], [], set()
        selected = self._pattern_candidates(user_utterance, tokens)
        unresolved = self._apply_stages(selected, matches, occupied, attempts_log)
        if not matches:
            self._token_fallback(tokens, matches, occupied, attempts_log)

        slot_values = {slot: [] for slot in self.allowed_slots}
        for item in sorted(matches, key=lambda m: (m["start"], m["end"])):
            if item["value"] not in slot_values[item["slot"]]:
                slot_values[item["slot"]].append(item["value"])
        pairs = [f"{slot}: {value}" for slot in self.allowed_slots for value in slot_values[slot]]
        returned = "(" + ", ".join(pairs) + ")"
        self._write_log({"utterance": user_utterance, "slots": slot_values,
                         "matches": matches, "matching_attempts": attempts_log,
                         "unresolved_pattern_candidates": [
                             {k: v for k, v in item.items() if k not in {"allow_special", "attempts"}}
                             for item in unresolved
                         ],
                         "matcher_errors": self._errors, "returned": returned})
        return returned

    def _write_log(self, record: dict[str, Any]) -> None:
        log_path = Path(self.interaction["log_path"])
        if not log_path.is_absolute():
            log_path = self.base_dir / log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
