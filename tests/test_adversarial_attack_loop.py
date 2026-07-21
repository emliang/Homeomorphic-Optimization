import torch

from homopt.experiments.adversarial.attack import PGDAttack


class _TinyClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(4, 2),
        )

    def forward(self, x):
        return self.net(x)


def test_pgd_attack_generate_supports_gd_and_adam():
    model = _TinyClassifier()
    device = torch.device("cpu")
    attack = PGDAttack(model, device=device, random_start=False, hom=False)
    images = torch.zeros(2, 1, 2, 2)
    labels = torch.tensor([0, 1], dtype=torch.long)

    adv_gd = attack.generate(images, labels, eps=0.1, alpha=0.05, iters=2, norm=2, optimizer_name="gd")
    adv_adam = attack.generate(images, labels, eps=0.1, alpha=0.05, iters=2, norm=2, optimizer_name="adam")

    assert adv_gd.shape == images.shape
    assert adv_adam.shape == images.shape
    assert torch.isfinite(adv_gd).all()
    assert torch.isfinite(adv_adam).all()
