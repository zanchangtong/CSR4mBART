import math

import torch

mask_span_distribution = None
_lambda = 3.5 # 平均长度是3.5
lambda_to_the_k = 1
e_to_the_minus_lambda = math.exp(-_lambda)
k_factorial = 1
ps = []
for k in range(0, 128):
    ps.append(e_to_the_minus_lambda * lambda_to_the_k / k_factorial)
    lambda_to_the_k *= _lambda
    k_factorial *= k + 1
    if ps[-1] < 0.0000001:
        break
num_to_mask=50 #  = int(math.ceil(is_word_start.float().sum() * p))
ps = torch.FloatTensor(ps)
mask_span_distribution = torch.distributions.Categorical(ps)
lengths = mask_span_distribution.sample(sample_shape=(num_to_mask,))
#print(lengths)
cum_length = torch.cumsum(lengths, 0)
#print(cum_length)
# Make sure we have enough to mask
while cum_length[-1] < num_to_mask:
    lengths = torch.cat(
        [
            lengths,
            mask_span_distribution.sample(sample_shape=(num_to_mask,)),
        ],
        dim=0,
    )
    cum_length = torch.cumsum(lengths, 0)
print(lengths)
print(cum_length)

# Trim to masking budget
i = 0
while cum_length[i] < num_to_mask:
    i += 1
lengths[i] = num_to_mask - (0 if i == 0 else cum_length[i - 1])
num_to_mask = i + 1
lengths = lengths[:num_to_mask]

lengths = lengths[lengths > 0]
num_inserts = num_to_mask - lengths.size(0)
num_to_mask -= num_inserts

#if num_to_mask == 0:
#    return self.add_insertion_noise(source, num_inserts / source.size(0))

assert (lengths > 0).all()
print(lengths)



