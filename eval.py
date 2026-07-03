import torch.utils.data
import torch
import torch.nn.functional as F
import os
import torchvision.utils as vutils
from utils.losses import SensoyEDLLoss
from sklearn.metrics import cohen_kappa_score
from utils.parser import get_parser_with_args
from utils.helpers import get_test_loaders, load_model
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

# The Evaluation Methods in our paper are slightly different from this file.
# In our paper, we use the evaluation methods in train.py. specifically, batch size is considered.
# And the evaluation methods in this file usually produce higher numerical indicators.

parser, metadata = get_parser_with_args()
opt = parser.parse_args()

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

test_loader = get_test_loaders(opt)

# Load trained model
model = load_model(opt, dev)

path = "./tmp/best_model.pt"      # change if needed

model = torch.load(path, map_location=dev)
model = model.to(dev)

edl_criterion = SensoyEDLLoss(
    num_classes=2,
    annealing_step=opt.epochs
)

total_edl_loss = 0.0

os.makedirs("results", exist_ok=True)

c_matrix = {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}
total_evidence = 0.0
total_uncertainty = 0.0
num_batches = 0

all_labels = []
all_preds = []
model.eval()

with torch.no_grad():
    tbar = tqdm(test_loader)
    for batch_img1, batch_img2, labels in tbar:

        batch_img1 = batch_img1.float().to(dev)
        batch_img2 = batch_img2.float().to(dev)
        labels = labels.long().to(dev)

 # Model outputs
        cd_preds = model(batch_img1, batch_img2)

# Final logits
        final_logits = cd_preds[-1]
        edl_loss = edl_criterion(
            final_logits,
            labels,
            epoch_num=opt.epochs
        )

        total_edl_loss += edl_loss.item()

# ---------- EDL ----------
        evidence = F.softplus(final_logits)

        alpha = evidence + 1

        probability = alpha / alpha.sum(dim=1, keepdim=True)

        uncertainty = 2.0 / alpha.sum(dim=1, keepdim=True)
        # Prediction mask
        prediction = torch.argmax(final_logits, dim=1, keepdim=True).float()

# Ground truth mask
        ground_truth = labels.unsqueeze(1).float()

# Change probability
        change_probability = probability[:, 1:2]

# Change evidence
        change_evidence = evidence[:, 1:2]

# Error map
        error = (prediction != ground_truth).float()
        num_batches += 1
        vutils.save_image(
           prediction,
           f"results/prediction_{num_batches}.png",
           normalize=True
        )

        vutils.save_image(
            ground_truth,
            f"results/ground_truth_{num_batches}.png",
            normalize=True
        )

        vutils.save_image(
            change_probability,
            f"results/probability_{num_batches}.png",
            normalize=True
        )

        vutils.save_image(
           uncertainty,
           f"results/uncertainty_{num_batches}.png",
           normalize=True
        )

        vutils.save_image(
            change_evidence,
            f"results/evidence_{num_batches}.png",
            normalize=True
        )

        vutils.save_image(
            error,
            f"results/error_{num_batches}.png",
            normalize=True
)

        total_evidence += evidence.mean().item()
        total_uncertainty += uncertainty.mean().item()
# -------------------------

        _, cd_preds = torch.max(final_logits, 1)

        tn, fp, fn, tp = confusion_matrix(labels.data.cpu().numpy().flatten(),
                        cd_preds.data.cpu().numpy().flatten()).ravel()

        c_matrix['tn'] += tn
        c_matrix['fp'] += fp
        c_matrix['fn'] += fn
        c_matrix['tp'] += tp
        all_labels.extend(labels.cpu().numpy().flatten())
        all_preds.extend(cd_preds.cpu().numpy().flatten())

tn, fp, fn, tp = c_matrix['tn'], c_matrix['fp'], c_matrix['fn'], c_matrix['tp']
OA = (tp + tn) / (tp + tn + fp + fn)
P = tp / (tp + fp) if (tp + fp) > 0 else 0.0
R = tp / (tp + fn) if (tp + fn) > 0 else 0.0
F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
kappa = cohen_kappa_score(all_labels, all_preds)

mean_evidence = total_evidence / num_batches

mean_uncertainty = total_uncertainty / num_batches
print(f"Precision        : {P:.4f}")
print(f"Recall           : {R:.4f}")
print(f"F1 Score         : {F1:.4f}")
print(f"Cohen Kappa      : {kappa:.4f}")
print(f"Mean Evidence    : {mean_evidence:.4f}")
print(f"Mean Uncertainty : {mean_uncertainty:.4f}")
print(f"Overall Accuracy: {OA:.4f}")
print(f"Test EDL Loss    : {total_edl_loss / num_batches:.4f}")