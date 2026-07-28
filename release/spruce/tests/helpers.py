import re


class WhitespaceTokenizer:
    """Small offset-preserving tokenizer for package tests."""

    def __init__(self):
        self.vocabulary = {}
        self.pad_token_id = None
        self.eos_token_id = 0

    def _tokens(self, text):
        return list(re.finditer(r"\w+|[^\w\s]", text.lower()))

    def __call__(
            self, text, return_offsets_mapping=False,
            add_special_tokens=False, **_kwargs):
        del add_special_tokens
        matches = self._tokens(text)
        ids = []
        for match in matches:
            token = match.group(0)
            if token not in self.vocabulary:
                self.vocabulary[token] = len(self.vocabulary) + 1
            ids.append(self.vocabulary[token])
        result = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = [
                (match.start(), match.end()) for match in matches
            ]
        return result

    def apply_chat_template(
            self, messages, tokenize=False, add_generation_prompt=True):
        assert not tokenize
        rendered = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered
