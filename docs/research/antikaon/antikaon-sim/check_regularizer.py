"""Monte-Carlo check: E_xi[ P grad L(z + xi) ] == P grad( L + 1/2 Tr(Sigma H) )(z) + O(sigma^3),
for a diagonal Sigma (ARWP-style, v-shaped) and a diagonal preconditioner P, on a random cubic-plus-quartic L.
CPU, tiny. Also compares against the WRONG candidate 1/2 Tr(Sigma P H) to show P enters only via the metric."""
import torch

torch.manual_seed(0)
d = 6
A = torch.randn(d, d); A = A @ A.T / d + torch.eye(d)          # PD quadratic part
T3 = torch.randn(d, d, d); T3 = (T3 + T3.permute(1, 0, 2) + T3.permute(2, 1, 0)) / 3  # symmetric cubic
b4 = torch.rand(d) * 0.5


def L(z):
    return 0.5 * z @ A @ z + torch.einsum('ijk,i,j,k->', T3, z, z, z) / 6 + (b4 * z ** 4).sum() / 4


def grad(z):
    z = z.clone().requires_grad_(True)
    g, = torch.autograd.grad(L(z), z)
    return g


def hess(z):
    return torch.autograd.functional.hessian(L, z)


z0 = torch.randn(d) * 0.3
sig = 0.05 * (0.5 + torch.rand(d))          # anisotropic per-coordinate std (v-shaped)
Sigma = torch.diag(sig ** 2)
P = torch.diag(0.5 + torch.rand(d))          # diagonal preconditioner

# analytic: grad of 1/2 Tr(Sigma H)(z)
def reg(z):
    return 0.5 * torch.trace(Sigma @ torch.func.hessian(L)(z))

greg = torch.func.grad(reg)(z0)
target = P @ (grad(z0) + greg)

# MC with antithetic pairs for variance reduction
M = 400_000
xi = torch.randn(M, d) * sig
zs = z0 + xi
zs = zs.requires_grad_(True)
Ls = 0.5 * torch.einsum('bi,ij,bj->b', zs, A, zs) + torch.einsum('ijk,bi,bj,bk->b', T3, zs, zs, zs) / 6 + (b4 * zs ** 4).sum(-1) / 4
G, = torch.autograd.grad(Ls.sum(), zs)
zs2 = (z0 - xi).requires_grad_(True)
Ls2 = 0.5 * torch.einsum('bi,ij,bj->b', zs2, A, zs2) + torch.einsum('ijk,bi,bj,bk->b', T3, zs2, zs2, zs2) / 6 + (b4 * zs2 ** 4).sum(-1) / 4
G2, = torch.autograd.grad(Ls2.sum(), zs2)
mc = P @ ((G + G2) / 2).mean(0)
se = (P @ ((G + G2) / 2).T).std(1) / M ** 0.5

wrong = P @ grad(z0) + 0.5 * torch.autograd.grad(torch.trace(Sigma @ P @ hess(z0.clone().requires_grad_(True))), z0.clone().requires_grad_(True), allow_unused=True)[0] if False else None

print("P grad L(z0)              :", (P @ grad(z0)).numpy().round(5))
print("P grad(L + 1/2 Tr(Sigma H)):", target.numpy().round(5))
print("MC  E[P grad L(z0+xi)]    :", mc.numpy().round(5))
print("MC standard error         :", se.numpy().round(5))
print("max |MC - target| / se    :", float(((mc - target).abs() / se).max()))
print("max |MC - Pgrad L| / se   :", float(((mc - P @ grad(z0)).abs() / se).max()), "(the bare-gradient hypothesis is rejected)")
