import torch
import ipdb

class ResidualTanhGatedRMSNorm(torch.nn.Module):
    def __init__(self):
        super(ResidualTanhGatedRMSNorm, self).__init__()

    def forward(self, x, x_res, gate, eps=1e-6):
        # Convert to fp32 for precision
        x_res_fp32 = x_res.float()

        # Compute RMS
        mean_square = x_res_fp32.pow(2).mean(-1, keepdim=True)
        scale = torch.rsqrt(mean_square + eps)

        # tanh_gate = torch.tanh(gate).unsqueeze(1) # TODO
        tanh_gate = torch.tanh(gate)

        # # Normalize and apply gated scaling643
        x_normed = x_res_fp32 * scale * tanh_gate

        # # TODO Split computation along last dimension to reduce memory
        # chunk_size = 512  # Adjust based on available memory (must divide 3072)
        # x_chunks = x_res_fp32.split(chunk_size, dim=-1)
        # gate_chunks = tanh_gate.split(chunk_size, dim=-1)
        # # Process chunks sequentially
        # normed_chunks = []
        # for x_chunk, gate_chunk in zip(x_chunks, gate_chunks):
        #     normed_chunks.append(x_chunk * scale * gate_chunk)
        # # Reconstruct full tensor
        # x_normed = torch.cat(normed_chunks, dim=-1)

        # Apply residual connection
        output = x + x_normed.type_as(x)

        return output


# def residual_tanh_gated_rmsnorm(x, x_res, gate, eps=1e-6):
#     return ResidualTanhGatedRMSNorm.apply(x, x_res, gate, eps)

def residual_tanh_gated_rmsnorm(x, x_res, gate, eps=1e-6):
    norm = ResidualTanhGatedRMSNorm()
    return norm.forward(x, x_res, gate, eps)
