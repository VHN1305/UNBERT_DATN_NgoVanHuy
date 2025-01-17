from typing import List, Tuple, Dict, Any
import torch
import torch.nn as nn
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# from .configuration_bert import BertConfig
from .unbert import UNBertModel
from transformers import BertConfig


class UNBERT(nn.Module):
    def __init__(
        self,
        pretrained: str,
        level_sate: str = "both",
        news_mode: str = "nseg",
        max_len: int = None,
    ) -> None:
        super(UNBERT, self).__init__()
        self._pretrained = pretrained
        self._level_state = level_sate
        self._news_mode = news_mode
        self._config = BertConfig.from_pretrained(self._pretrained)
        self._model = UNBertModel.from_pretrained(self._pretrained, config=self._config)
        if self._level_state == "both":
            self._dense = nn.Linear(self._config.hidden_size * 2, 2)
        else:
            self._dense = nn.Linear(self._config.hidden_size, 2)

        if self._news_mode == "attention":
            assert max_len is not None
            self.att = nn.Sequential(
                nn.Linear(max_len * self._config.hidden_size, 128),
                nn.Sigmoid(),
                nn.Linear(128, max_len),
            )
        else:
            self.att = None

    def forward(
        self,
        input_ids: torch.Tensor,
        input_mask: torch.Tensor = None,
        segment_ids: torch.Tensor = None,
        news_segment_ids: torch.Tensor = None,
        sentence_ids: torch.Tensor = None,
        sentence_mask: torch.Tensor = None,
        sentence_segment_ids: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self._model(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=segment_ids,
            news_segment_ids=news_segment_ids,
            sentence_input=(sentence_ids, sentence_mask, sentence_segment_ids),
            news_mode=self._news_mode,
            att_mapping=self.att,
        )
        word_hidden = output[0][:, 0, :]
        sentence_hidden = output[1][:, 0, :]
        if self._level_state == "word":
            hidden = word_hidden
        elif self._level_state == "news":
            hidden = sentence_hidden
        else:
            hidden = torch.cat([word_hidden, sentence_hidden], -1)
        score = self._dense(hidden).squeeze(-1)
        return score


if __name__ == "__main__":
    model = UNBERT("BAAI/bge-large-en-v1.5")
    print(model)
