(c) Alex Spirin 2023
#rewritten PDF color transfer from https://github.com/pengbo-learn/python-color-transfer in pytorch for almost x50 speedup
def torch_interp1d(x, xp, fp):
    # print('x, xp, fp.device')
    # print(x.device, xp.device, fp.device)
    """
    Performs 1D linear interpolation.

    Args:
    x (Tensor): The x-coordinates at which to evaluate the interpolated values.
    xp (Tensor): The x-coordinates of the data points.
    fp (Tensor): The y-coordinates of the data points, same length as xp.

    Returns:
    Tensor: Interpolated values for each element of x.
    """
    idxs = torch.searchsorted(xp, x)
    idxs = idxs.clamp(1, len(xp) - 1).cuda()
    left = xp[idxs - 1]
    right = xp[idxs]
    alpha = (x - left) / (right - left)
    print('alpha.device, fp.device, idxd.device')
    print(alpha.device, fp.device, idxs.device)
    fp=fp.cuda()
    return fp[idxs - 1] + alpha * (fp[idxs] - fp[idxs - 1])

def pdf_transfer_nd_torch(self, arr_in=None, arr_ref=None, step_size=1):
    # print('arr_in.device, arr_ref.device')
    # print(arr_in.device, arr_ref.device)
    """Apply n-dim probability density function transfer in PyTorch.

    Args:
        arr_in: shape=(n, x), PyTorch tensor.
        arr_ref: shape=(n, x), PyTorch tensor.
        step_size: arr = arr + step_size * delta_arr.
    Returns:
        arr_out: shape=(n, x), PyTorch tensor.
    """
    # Initialize the output tensor
    arr_out = arr_in.clone()

    # Loop through rotation matrices
    for rotation_matrix in self.rotation_matrices_torch:
        # Rotate input and reference arrays
        rot_arr_in = torch.matmul(rotation_matrix, arr_out)
        rot_arr_ref = torch.matmul(rotation_matrix, arr_ref)

        # Initialize the rotated output array
        rot_arr_out = torch.zeros_like(rot_arr_in)

        # Loop over the first dimension
        for i in range(rot_arr_out.shape[0]):
            # Apply 1D PDF transfer (assuming _pdf_transfer_1d is adapted for PyTorch)
            rot_arr_out[i] = self._pdf_transfer_1d_torch(rot_arr_in[i], rot_arr_ref[i])

        # Calculate the delta array
        rot_delta_arr = rot_arr_out - rot_arr_in
        delta_arr = torch.matmul(rotation_matrix.transpose(0, 1), rot_delta_arr)

        # Update the output array
        arr_out = arr_out + step_size * delta_arr

    return arr_out

import torch

def _pdf_transfer_1d_torch(self, arr_in=None, arr_ref=None, n=300):
    # print('arr_in.device, arr_ref.device')
    # print(arr_in.device, arr_ref.device)
    """Apply 1-dim probability density function transfer using PyTorch.

    Args:
        arr_in: 1d PyTorch tensor input array.
        arr_ref: 1d PyTorch tensor reference array.
        n: discretization num of distribution of image's pixels.
    Returns:
        arr_out: transferred input tensor.
    """
    arr = torch.cat((arr_in, arr_ref))
    min_v = torch.min(arr) - self.eps
    max_v = torch.max(arr) + self.eps
    xs = torch.linspace(min_v, max_v, steps=n+1).to(arr.device)

    # Compute histograms
    hist_in = torch.histc(arr_in, bins=n, min=min_v, max=max_v)
    hist_ref = torch.histc(arr_ref, bins=n, min=min_v, max=max_v)
    xs = xs[:-1]

    # Compute cumulative distributions
    cum_in = torch.cumsum(hist_in, dim=0)
    cum_ref = torch.cumsum(hist_ref, dim=0)
    d_in = cum_in / cum_in[-1]
    d_ref = cum_ref / cum_ref[-1]

    # Transfer function
    t_d_in = torch_interp1d(d_in, d_ref, xs)
    t_d_in[d_in <= d_ref[0]] = min_v
    t_d_in[d_in >= d_ref[-1]] = max_v
    arr_out = torch_interp1d(arr_in, xs, t_d_in)

    return arr_out

import torch

def pdf_transfer_torch(self, img_arr_in=None, img_arr_ref=None, regrain=False):
    """Apply probability density function transfer using PyTorch.

    img_o = t(img_i) so that f_{t(img_i)}(r, g, b) = f_{img_r}(r, g, b),
    where f_{img}(r, g, b) is the probability density function of img's rgb values.

    Args:
        img_arr_in: BGR PyTorch tensor of input image.
        img_arr_ref: BGR PyTorch tensor of reference image.
    Returns:
        img_arr_out: Transferred BGR PyTorch tensor of input image.
    """

    # Ensure input is a PyTorch tensor
    img_arr_in = torch.tensor(img_arr_in, dtype=torch.float32, device='cuda') / 255.0
    img_arr_ref = torch.tensor(img_arr_ref, dtype=torch.float32, device='cuda') / 255.0
    # print('img_arr_in.device, img_arr_ref.device')
    # print(img_arr_in.device, img_arr_ref.device)
    # Reshape (h, w, c) to (c, h*w)
    [h, w, c] = img_arr_in.shape
    reshape_arr_in = img_arr_in.view(-1, c).permute(1, 0)
    reshape_arr_ref = img_arr_ref.view(-1, c).permute(1, 0)

    # PDF transfer
    reshape_arr_out = self.pdf_transfer_nd_torch(arr_in=reshape_arr_in,
                                                 arr_ref=reshape_arr_ref)

    # Reshape (c, h*w) to (h, w, c)
    reshape_arr_out.clamp_(0, 1)  # Ensure values are between 0 and 1
    reshape_arr_out = (255.0 * reshape_arr_out).to(torch.uint8)
    img_arr_out = reshape_arr_out.permute(1, 0).view(h, w, c)

    if regrain:
        img_arr_in = (255.0 * img_arr_in).to(torch.uint8)
        img_arr_out = self.RG.regrain(img_arr_in=img_arr_in,
                                      img_arr_col=img_arr_out)

    return img_arr_out.detach().cpu().numpy()
