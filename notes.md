# Note

## Arguments

ada_reso_skip           = True
use_reinforce           = False
multi_models            = True
policy_also_backbone    = True
offline_lstm_last       = False
offline_lstm_all        = False
real_scsampler          = False
random_policy           = False
all_policy              = False
save_meta               = False
save_all_preds          = False
online_policy           = True
frame_independent       = False
uniform_cross_entropy   = False
policy_input_offset     = 3
action_dim              = 7
reso_dim                = 4
skip_list               = [1, 2, 4]
ada_crop_list           = [1, 1, 1, 1]
uniform_loss_weight     = 3

## Variables

input_list --> list of inputs at different resolutions (224, 168, 112, 84)
lite_j_list --> list of light weight features from the scanning model (MobileNetV2)
r_all --> sampling vector of shape (B, T, 7)

## Workflow

### forward()

Main forward function

```python
input_list = kwargs["input"]
batch_size = input_list[0].shape[0]

lite_j_list, r_all = self.get_lite_j_and_r(input_list, self.using_online_policy(), kwargs["tau"])
feat_out_list, base_out_list, ind_list = self.get_feat_and_pred(input_list, r_all, tau=kwargs["tau"])
base_out_list.append(torch.stack(lite_j_list, dim=1))
output = self.combine_logits(r_all, base_out_list, ind_list)

return output.squeeze(1), r_all, None, torch.stack(base_out_list, dim=1)
```

### get_lite_j_and_r()

Get the lite weight feature lite_j_list and sampling vector r_all

```python
feat_lite = MobileNetV2(...)
hx, cx = 0, 0
old_hx, old_r_t = None, None
remain_skip_vector = zeros(batch_size, 1)
for t in range(self.time_steps):
    hx, cx = rnn(feat_lite[:, t], (hx, cx))
    feat_t = hx
    p_t = log(softmax(linear(feat_t))).clamp(min=1e-8)
    j_t = lite_fc(feat_t)
    lite_j_list.append(j_t)

    r_t = cat([gumbel_softmax(p_t])

    if old_hx is not None:
        ...
        hx = old_hx*take_old + hx*take_curr
        r_t = old_r_t*take_old + r_t*take_curr

    update remaining_skip_vector

    old_dx, old_r_t = hx, r_t
    r_list.append(r_t)
    remain_skip_vector = (remain_skip_vector - 1).clamp(0)

return lite_j_list, stack(r_list)
```

### combine_logits()

Combine the r_t with pred_tensor

## Loss

```python
r_loss = [gflops_resnet_224, gflops_resnet_168, gflops_resnet_112, 0, 0, 0, 0]
loss = sum(r.mean(dim=[0, 1]) * r_loss)

reso_skip_vec = r.mean(dim=[0, 1])
usage_bias = reso_skip_vec - mean(reso_skip_vec)
uniform_loss = norm(usage_bias, p=2) * uniform_loss_weight

loss += uniform_loss
```
