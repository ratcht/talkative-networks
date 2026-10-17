# Probing the communication channel

Goal: figure out if communication is worthwhile. Linear probes (Alain & Bengio 2016)
on frozen features, fit on `ensemble_indices`, scored on `val_indices`. Probes never
touch the model.

Three questions, in order:

## 1. Existence — do the experts have complementary info?

`probe(concat h)` vs `probe(best single h_i)`, where `h_i` = mean-pooled late feats.
The gap is the marginal information — the only thing communication could ever deliver.
If concat ≈ best single, experts are redundant and communication is dead on arrival.

## 2. Delivery — does the channel transmit it?

`probe(concat(h_i, c_i))` vs `probe(h_i)`, where `c_i` = message received off the bus.
Note `probe(c_i)` alone is the wrong test: c could score below h and still add
complementary bits. If gap 1 is positive but gap 2 is ~0, the channel is the
bottleneck (encoder -> 64d mean-pooled value -> ~uniform attention mixing, which also
dilutes with the receivers own V_i).

## 3. Usability — can the receiver exploit what arrives?

Not answerable by probes: messages are injected as a bias at layer2.0 of a frozen
backbone. Only the end-to-end comparison (communicative vs k=0 control) answers this.

## Caveats

- V_i is a linear map of h_i, so the channel can never carry more than h contains.
  The depth sweep is context, not the main event.
- Prior: 3 snapshot experts trained on the same 25k images probably have limited
  diversity. If gap 1 is ~0 the fix is a more diverse expert pool, not bus machinery.
