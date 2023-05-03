import einops
import torch.nn as nn
import torch.nn.functional as F
import torch
from ..utils import getter
import torchvision
from .baseline7_utils import *
import json
from functorch import vmap

__all__ = ["GazeBaseline7"]


class GazeBaseline7(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        with open(config["vocab_path"], "r") as f:
            self.vocab = json.load(f)
        self.vocab_size = config["vocab_size"]
        self.max_number_sent = config["max_number_sent"]
        self.img_encoder = ImageEncoder(config)
        self.fixation_encoder = FixationEncoderPE2D(config)
        self.image_fixation_fusion = ImageFixationFuser(config)
        # multihead attention
        self.caption_embed = CaptionEmbedding(config)
        self.att_capin = nn.MultiheadAttention(
            config["hidden_size"],
            config["num_attention_heads"],
            dropout=config["attention_probs_dropout_prob"],
            batch_first=True,
        )

        self.transformer_decoder_layer = nn.TransformerDecoderLayer(
            config["hidden_size"],
            config["num_attention_heads"],
            dim_feedforward=config["intermediate_size"],
            dropout=config["hidden_dropout_prob"],
            batch_first=True,
        )
        self.norm_capout = nn.LayerNorm(config["hidden_size"])
        self.transformer_decoder = nn.TransformerDecoder(
            self.transformer_decoder_layer,
            config["num_hidden_layers"],
            norm=self.norm_capout,
        )
        # self.decoder_word = nn.Sequential(nn.Linear(config["hidden_size"], self.vocab_size), nn.LogSoftmax(dim=-1))
        self.decoder_word = nn.Linear(
            config["hidden_size"], self.vocab_size
        )  # if use cross entropy loss from pytorch, no need for log softmax

        self.pe = PositionalEncoding(
            config["hidden_size"], dropout=config["hidden_dropout_prob"]
        )
        self.double_pe = DoublePE(config)
        # self.learnable_pe = nn.Embedding(self.max_number_sent,400* config["hidden_size"])

        self.number_prediction = nn.Sequential(nn.Linear(config["hidden_size"], config["hidden_size"]//2),nn.ReLU(), nn.Linear(config["hidden_size"]//2, 3))
        self.max_sent_len = 110 # max full transcript is 109, min is 1


    def forward(self, img, fixation, fix_masks, captions, cap_masks):
        # img torch.Size([1, 3, 224, 224]) fixation torch.Size([1, 400, 3]) fix_masks torch.Size([1, 400, 1]) captions torch.Size([1, 3, 50]) cap_masks torch.Size([1, 3, 50])
        # because it is fixation, i keep the original shape: full size x 3, instead of split for each sentences

        # embedding to create "memory" for decoder
        # embedding for image
        img_features = self.img_encoder(img)  # torch.Size([1, leni, 512])

        # embedding for fixation
        fix_feature = self.fixation_encoder(
            fixation, fix_masks
        )  # torch.Size([1, 400, 512])

        # fusing between img and fixation
        fused_img_fix = self.image_fixation_fusion(
            img_features, fix_feature, length=captions.shape[1]
        )  # torch.Size([ 3, 400+leni, 512])
        num = self.number_prediction(img_features.mean(1))  # torch.Size([1, 3])
        # learnable_pe = self.learnable_pe.weight.unsqueeze(0).view(self.max_number_sent, 400, -1)
        # fused_img_fix = fused_img_fix + learnable_pe[:captions.shape[1]]

        # embedding for caption
        cap_feature, cap_masks_tril = self.caption_embed(
            captions, cap_masks
        )  # torch.Size([3, 50, 512]) torch.Size([3*numhead, 50, 50])
        output = self.transformer_decoder(
            cap_feature, fused_img_fix, tgt_mask=cap_masks_tril
        )
        output_sent = self.decoder_word(output)  # torch.Size([3, 50, vocab_size])
        # torch.Size([3, 50, 512])
        tmp = torch.argmax(output_sent, dim=2)
        return output_sent, num

    def build_loss(self, pred, target, mask):
        # torch.Size([max_number_sent/gt_length, 50, vocab_size]) torch.Size([1, gt_length, 50]) torch.Size([1, gt_length, 50])
        # target = einops.rearrange(target, 'b s l -> (b s) l') # torch.Size([3, 50])
        # mask = einops.rearrange(mask, 'b s l -> (b s) l') # torch.Size([3, 50])
        # one_hot = torch.nn.functional.one_hot(target, self.config['vocab_size'])
        # gt_number_sent = target.shape[0]
        # output = -(one_hot * pred[:gt_number_sent] * mask[:, :, None]).sum(2).sum(1) / (mask.sum(1) + 1e-6)
        # return output.mean()
        target = einops.rearrange(target, "b s l -> (b s) l")  # torch.Size([3, 50])
        mask = einops.rearrange(mask, "b s l -> (b s) l")  # torch.Size([3, 50])
        gt_number_sent = target.shape[0]
        N, T, V = pred[:gt_number_sent].shape

        x_flat = pred[:gt_number_sent].reshape(N * T, V)
        y_flat = target.reshape(N * T)
        mask_flat = mask.reshape(N * T)

        loss = torch.nn.functional.cross_entropy(
            x_flat, y_flat, reduction="none", label_smoothing=0.1
        )
        loss = torch.mul(loss, mask_flat)
        loss = torch.mean(loss)
        return loss

    def generate_greedy(self, img, fixation, fix_masks):
        # generate the caption with beam search
        # beam search implementation here
        # embedding to create "memory" for decoder
        # embedding for image

        img_features = self.img_encoder(img)  # torch.Size([1, leni, 512])

        # embedding for fixation
        fix_feature = self.fixation_encoder(
            fixation, fix_masks
        )  # torch.Size([1, 400, 512])

        # fusing between img and fixation
        fused_img_fix = self.image_fixation_fusion(
            img_features, fix_feature, self.max_number_sent
        )  # torch.Size([ maxnumsent, 400 + leni, 512])
        num = self.number_prediction(img_features.mean(1))
        # learnable_pe = self.learnable_pe.weight.unsqueeze(0).view(self.max_number_sent, 400, -1)
        # fused_img_fix = fused_img_fix + learnable_pe
        # decoding
        # start with <sos> token
        # torch.Size([1, 1])
        cap_output = torch.tensor(
            [self.vocab["word2idx"]["<SOS>"]], device=fused_img_fix.device
        ).unsqueeze(0)
        cap_output = einops.repeat(
            cap_output, "b s ->() (l b) (s k)", l=self.max_number_sent, k=self.max_sent_len-1
        )  # torch.Size([1, max_num_sent, 50])
        # mask
        cap_masks = torch.zeros(
            (1, cap_output.shape[1], self.max_sent_len-1),
            device=fused_img_fix.device,
            dtype=torch.float32,
        )  # torch.Size([1, max_num_sent, 50])
        probs_cap = None
        word_cap = None
        cap_masks[..., 0] = 1.0
        for i in range(self.max_sent_len-1):
            cap_output = cap_output.clone()
            if i != 0:
                cap_output[..., i] = next_token
                cap_masks[..., i] = 1.0
            cap_feature, cap_masks_tril = self.caption_embed(cap_output, cap_masks)
            output = self.transformer_decoder(
                cap_feature, fused_img_fix, tgt_mask=cap_masks_tril
            )
            output_sent = self.decoder_word(
                output
            )  # torch.Size([max_num_sent, 50, vocab_size])

            next_token = torch.argmax(
                output_sent[:, i], dim=1
            )  # torch.Size([max_num_sent, 50])
            if probs_cap is None:
                probs_cap = output_sent[:, i].unsqueeze(1)
            else:
                probs_cap = torch.cat(
                    (probs_cap, output_sent[:, i].unsqueeze(1)), dim=1
                )
            if word_cap is None:
                word_cap = next_token.unsqueeze(1)
            else:
                word_cap = torch.cat((word_cap, next_token.unsqueeze(1)), dim=1)
        return word_cap, probs_cap, num

    def beam_search(self, img, fixation, fix_masks, beam_size = 3):
        """Function for beam search

        Args:
            x (_type_): x does not matter, basically a dummy input
        """        
        # decoding
        # start with <sos> token
        # torch.Size([1, 1])
        img_features = self.img_encoder(img)  # torch.Size([1, leni, 512])

        # embedding for fixation
        fix_feature = self.fixation_encoder(
            fixation, fix_masks
        )  # torch.Size([1, 400, 512])

        # fusing between img and fixation
        fused_img_fix = self.image_fixation_fusion(
            img_features, fix_feature, self.max_number_sent
        )  # torch.Size([ maxnumsent, 400 + leni, 512])
        num = self.number_prediction(img_features.mean(1))

        bs = 2 # max_num_sent
        max_len = 50
        cap_output = torch.tensor(
            [self.vocab["SOS"]], device=img.device
        ).unsqueeze(0)
        cap_output = einops.repeat(
            cap_output, "b s ->() (l b) (s k)", l=2*beam_size, k=max_len
        )  # torch.Size([1, max_num_sent*beam_size = 3, 50])
        # mask
        cap_masks = torch.zeros(
            (1, cap_output.shape[1], max_len),
            device=img.device,
            dtype=torch.float32,
        )  # torch.Size([1, max_num_sent*beam_size = 3, 50])
        
        probs_cap = None
        word_cap = None
        beam_scores = img.new(bs, beam_size).fill_(0)
        beam_scores[:, 1:] = -1e9 # to mask the first top k, to pick only top k=1 at the first step
        beam_scores = beam_scores.view(-1)
        cap_masks[..., 0] = 1.0
        done = [False for _ in range(bs)]
        
        length_penalty = 1.0
        generated_hyps = [BeamHypotheses(beam_size, max_len, length_penalty, False) for _ in range(bs)]
        generated = cap_output.new(bs*beam_size, max_len).fill_(self.vocab["SOS"])
        for cur_len in range(max_len):
            # cap_output = generated.clone()
            if cur_len != 0:
                cap_output = cap_output[:, beam_idx, ...]
                cap_output[..., cur_len] = beam_words
                cap_masks[..., cur_len] = 1.0
            cap_feature, cap_masks_tril = self.caption_embed(cap_output, cap_masks)
            output = self.transformer_decoder(
                cap_feature, fused_img_fix, tgt_mask=cap_masks_tril
            )
            output_sent = self.decoder_word(
                output
            )  # torch.Size([max_num_sent, 50, vocab_size])
            # output_sent = self.pred(cap_output, cur_len) # torch.Size([max_num_sent, 50, vocab_size])
            tensor_we_care = output_sent[:, cur_len] # torch.Size([max_num_sent, vocab_size])
            # Set score to zero where EOS has been reached
                                                
            scores = F.log_softmax(tensor_we_care, dim=-1)       # (bs * beam_size, self.vocab_size)
            
            assert scores.size() == (bs * beam_size, self.vocab_size), f"{scores.size()} vs {(bs * beam_size, self.vocab_size)}"

            # select next words with scores
            _scores = scores + beam_scores[:, None].expand_as(scores)  # (bs * beam_size, self.vocab_size)
            _scores = _scores.view(bs, beam_size * self.vocab_size)            # (bs, beam_size * self.vocab_size)

            next_scores, next_words = torch.topk(_scores, 2 * beam_size, dim=1, largest=True, sorted=True)
            assert next_scores.size() == next_words.size() == (bs, 2 * beam_size)

            # next batch beam content
            # list of (bs * beam_size) tuple(next hypothesis score, next word, current position in the batch)
            next_batch_beam = []

            # for each sentence
            for sent_id in range(bs):

                # if we are done with this sentence
                done[sent_id] = done[sent_id] or generated_hyps[sent_id].is_done(next_scores[sent_id].max().item())
                if done[sent_id]:
                    next_batch_beam.extend([(0, self.vocab["PAD"], 0)] * beam_size)  # pad the batch
                    continue

                # next sentence beam content
                next_sent_beam = []

                # next words for this sentence
                for idx, value in zip(next_words[sent_id], next_scores[sent_id]):

                    # get beam and word IDs
                    beam_id = idx // self.vocab_size
                    word_id = idx % self.vocab_size

                    # end of sentence, or next word
                    if word_id == self.vocab["EOS"] or cur_len + 1 == max_len:
                        generated_hyps[sent_id].add(generated[sent_id * beam_size + beam_id, :cur_len].clone(), value.item())
                    else:
                        next_sent_beam.append((value, word_id, sent_id * beam_size + beam_id))

                    # the beam for next step is full
                    if len(next_sent_beam) == beam_size:
                        break

                # update next beam content
                assert len(next_sent_beam) == 0 if cur_len + 1 == max_len else beam_size
                if len(next_sent_beam) == 0:
                    next_sent_beam = [(0, self.vocab["PAD"], 0)] * beam_size  # pad the batch
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == beam_size * (sent_id + 1)

            # sanity check / prepare next batch
            assert len(next_batch_beam) == bs * beam_size
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_words = generated.new([x[1] for x in next_batch_beam])
            beam_idx = img.new([x[2] for x in next_batch_beam]).long()

            # re-order batch and internal states
            generated = generated[beam_idx, ... ]
            generated[..., cur_len] = beam_words
            # stop when we are done with each sentence
            if all(done):
                break

        tgt_len = img.new(bs).long()
        best = []

        for i, hypotheses in enumerate(generated_hyps):
            best_hyp = max(hypotheses.hyp, key=lambda x: x[0])[1]
            tgt_len[i] = len(best_hyp) + 1  # +1 for the <EOS> symbol
            best.append(best_hyp)

        # generate target batch
        decoded = img.new(tgt_len.max().item(), bs).fill_(self.vocab["PAD"])
        for i, hypo in enumerate(best):
            decoded[:tgt_len[i] - 1, i] = hypo
            decoded[tgt_len[i] - 1, i] = self.vocab["EOS"]

        # sanity check
        # assert (decoded == self.vocab["EOS"]).sum() >= 2 * bs, f'{(decoded == self.vocab["EOS"]).sum()} vs {2 * bs}'
        # if i use this beam search, i wont be able to compute the loss of the model
        return decoded, tgt_len, num
    