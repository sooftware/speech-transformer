
import torch
import torch.nn as nn
from torch import Tensor

from speech_transformer.decoder import SpeechTransformerDecoder


class BeamTransformerDecoder(nn.Module):
    def __init__(self, decoder: SpeechTransformerDecoder, batch_size: int, beam_size: int = 3) -> None:
        super(BeamTransformerDecoder, self).__init__()
        self.decoder = decoder
        self.beam_size = beam_size
        self.sos_id = decoder.sos_id
        self.pad_id = decoder.pad_id
        self.eos_id = decoder.eos_id
        self.ongoing_beams = None
        self.cumulative_ps = None
        self.finished = [[] for _ in range(batch_size)]
        self.finished_ps = [[] for _ in range(batch_size)]
        self.forward_step = decoder.forward_step
        self.use_cuda = True if torch.cuda.is_available() else False

    def _inflate(self, tensor: Tensor, n_repeat: int, dim: int) -> Tensor:
        repeat_dims = [1] * len(tensor.size())
        repeat_dims[dim] *= n_repeat

        return tensor.repeat(*repeat_dims)

    def _get_successor(
            self,
            current_ps: Tensor,
            current_vs: Tensor,
            finished_ids: tuple,
            num_successor: int,
            eos_count: int,
            k: int
    ) -> int:
        finished_batch_idx, finished_idx = finished_ids

        successor_ids = current_ps.topk(k + num_successor)[1]
        successor_idx = successor_ids[finished_batch_idx, -1]

        successor_p = current_ps[finished_batch_idx, successor_idx]
        successor_v = current_vs[finished_batch_idx, successor_idx]

        prev_status_idx = (successor_idx // k)
        prev_status = self.ongoing_beams[finished_batch_idx, prev_status_idx]
        prev_status = prev_status.view(-1)[:-1]

        successor = torch.cat([prev_status, successor_v.view(1)])

        if int(successor_v) == self.eos_id:
            self.finished[finished_batch_idx].append(successor)
            self.finished_ps[finished_batch_idx].append(successor_p)
            eos_count = self._get_successor(
                current_ps=current_ps,
                current_vs=current_vs,
                finished_ids=finished_ids,
                num_successor=num_successor + eos_count,
                eos_count=eos_count + 1,
                k=k,
            )

        else:
            self.ongoing_beams[finished_batch_idx, finished_idx] = successor
            self.cumulative_ps[finished_batch_idx, finished_idx] = successor_p

        return eos_count

    def _get_hypothesis(self):
        predictions = list()

        for batch_idx, batch in enumerate(self.finished):
            # if there is no terminated sentences, bring ongoing sentence which has the highest probability instead
            if len(batch) == 0:
                prob_batch = self.cumulative_ps[batch_idx]
                top_beam_idx = int(prob_batch.topk(1)[1])
                predictions.append(self.ongoing_beams[batch_idx, top_beam_idx])

            # bring highest probability sentence
            else:
                top_beam_idx = int(torch.FloatTensor(self.finished_ps[batch_idx]).topk(1)[1])
                predictions.append(self.finished[batch_idx][top_beam_idx])

        predictions = self._fill_sequence(predictions)
        return predictions

    def _is_all_finished(self, k: int) -> bool:
        for done in self.finished:
            if len(done) < k:
                return False

        return True

    def _fill_sequence(self, y_hats: list) -> Tensor:
        batch_size = len(y_hats)
        max_length = -1

        for y_hat in y_hats:
            if len(y_hat) > max_length:
                max_length = len(y_hat)

        matched = torch.zeros((batch_size, max_length), dtype=torch.long)

        for batch_idx, y_hat in enumerate(y_hats):
            matched[batch_idx, :len(y_hat)] = y_hat
            matched[batch_idx, len(y_hat):] = int(self.pad_id)

        return matched

    def forward(self, encoder_outputs: torch.FloatTensor, encoder_output_lengths: torch.FloatTensor):
        batch_size = encoder_outputs.size(0)

        decoder_inputs = torch.IntTensor(batch_size, self.decoder.max_length).fill_(self.sos_id).long()
        decoder_input_lengths = torch.IntTensor(batch_size).fill_(1)

        outputs = self.forward_step(
            decoder_inputs=decoder_inputs[:, :1],
            decoder_input_lengths=decoder_input_lengths,
            encoder_outputs=encoder_outputs,
            encoder_output_lengths=encoder_output_lengths,
            positional_encoding_length=1,
        )
        step_outputs = self.decoder.fc(outputs).log_softmax(dim=-1)
        self.cumulative_ps, self.ongoing_beams = step_outputs.topk(self.beam_size)

        self.ongoing_beams = self.ongoing_beams.view(batch_size * self.beam_size, 1)
        self.cumulative_ps = self.cumulative_ps.view(batch_size * self.beam_size, 1)

        decoder_inputs = torch.IntTensor(batch_size * self.beam_size, 1).fill_(self.sos_id)
        decoder_inputs = torch.cat((decoder_inputs, self.ongoing_beams), dim=-1)  # bsz * beam x 2

        encoder_dim = encoder_outputs.size(2)
        encoder_outputs = self._inflate(encoder_outputs, self.beam_size, dim=0)
        encoder_outputs = encoder_outputs.view(self.beam_size, batch_size, -1, encoder_dim)
        encoder_outputs = encoder_outputs.transpose(0, 1)
        encoder_outputs = encoder_outputs.reshape(batch_size * self.beam_size, -1, encoder_dim)

        encoder_output_lengths = encoder_output_lengths.unsqueeze(1).repeat(1, self.beam_size).view(-1)

        for di in range(2, self.decoder.max_length):
            if self._is_all_finished(self.beam_size):
                break

            decoder_input_lengths = torch.LongTensor(batch_size * self.beam_size).fill_(di)

            step_outputs = self.forward_step(
                decoder_inputs=decoder_inputs[:, :di],
                decoder_input_lengths=decoder_input_lengths,
                encoder_outputs=encoder_outputs,
                encoder_output_lengths=encoder_output_lengths,
                positional_encoding_length=di,
            )
            step_outputs = self.decoder.fc(step_outputs).log_softmax(dim=-1)

            step_outputs = step_outputs.view(batch_size, self.beam_size, -1, 10)
            current_ps, current_vs = step_outputs.topk(self.beam_size)

            # TODO: Check transformer's beam search
            current_ps = current_ps[:, :, -1, :]
            current_vs = current_vs[:, :, -1, :]

            self.cumulative_ps = self.cumulative_ps.view(batch_size, self.beam_size)
            self.ongoing_beams = self.ongoing_beams.view(batch_size, self.beam_size, -1)

            current_ps = (current_ps.permute(0, 2, 1) + self.cumulative_ps.unsqueeze(1)).permute(0, 2, 1)
            current_ps = current_ps.view(batch_size, self.beam_size ** 2)
            current_vs = current_vs.contiguous().view(batch_size, self.beam_size ** 2)

            self.cumulative_ps = self.cumulative_ps.view(batch_size, self.beam_size)
            self.ongoing_beams = self.ongoing_beams.view(batch_size, self.beam_size, -1)

            topk_current_ps, topk_status_ids = current_ps.topk(self.beam_size)
            prev_status_ids = (topk_status_ids // self.beam_size)

            topk_current_vs = torch.zeros((batch_size, self.beam_size), dtype=torch.long)
            prev_status = torch.zeros(self.ongoing_beams.size(), dtype=torch.long)

            for batch_idx, batch in enumerate(topk_status_ids):
                for idx, topk_status_idx in enumerate(batch):
                    topk_current_vs[batch_idx, idx] = current_vs[batch_idx, topk_status_idx]
                    prev_status[batch_idx, idx] = self.ongoing_beams[batch_idx, prev_status_ids[batch_idx, idx]]

            self.ongoing_beams = torch.cat([prev_status, topk_current_vs.unsqueeze(2)], dim=2)
            self.cumulative_ps = topk_current_ps

            if torch.any(topk_current_vs == self.eos_id):
                finished_ids = torch.where(topk_current_vs == self.eos_id)
                num_successors = [1] * batch_size

                for (batch_idx, idx) in zip(*finished_ids):
                    self.finished[batch_idx].append(self.ongoing_beams[batch_idx, idx])
                    self.finished_ps[batch_idx].append(self.cumulative_ps[batch_idx, idx])

                    if self.beam_size != 1:
                        eos_count = self._get_successor(
                            current_ps=current_ps,
                            current_vs=current_vs,
                            finished_ids=(batch_idx, idx),
                            num_successor=num_successors[batch_idx],
                            eos_count=1,
                            k=self.beam_size,
                        )
                        num_successors[batch_idx] += eos_count

            ongoing_beams = self.ongoing_beams.clone().view(batch_size * self.beam_size, -1)
            decoder_inputs = torch.cat((decoder_inputs, ongoing_beams[:, :-1]), dim=-1)

        return self._get_hypothesis()
