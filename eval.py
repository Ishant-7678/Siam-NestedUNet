import torch.utils.data
import torch
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score
from utils.parser import get_parser_with_args
from utils.helpers import get_test_loaders
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

# The Evaluation Methods in our paper are slightly different from this file.
# In our paper, we use the evaluation methods in train.py. specifically, batch size is considered.
# And the evaluation methods in this file usually produce higher numerical indicators.

parser, metadata = get_parser_with_args()
opt = parser.parse_args()

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

test_loader = get_test_loaders(opt)

path = 'weights/snunet-32.pt'   # the path of the model
model = torch.load(path)

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

# ---------- EDL ----------
        evidence = F.softplus(final_logits)

        alpha = evidence + 1

        probability = alpha / alpha.sum(dim=1, keepdim=True)

        uncertainty = 2.0 / alpha.sum(dim=1, keepdim=True)

        total_evidence += evidence.mean().item()
        total_uncertainty += uncertainty.mean().item()
        num_batches += 1
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
P = tp / (tp + fp)
R = tp / (tp + fn)
F1 = 2 * P * R / (R + P)
kappa = cohen_kappa_score(all_labels, all_preds)

mean_evidence = total_evidence / num_batches

mean_uncertainty = total_uncertainty / num_batches
print(f"Precision        : {P:.4f}")
print(f"Recall           : {R:.4f}")
print(f"F1 Score         : {F1:.4f}")
print(f"Cohen Kappa      : {kappa:.4f}")
print(f"Mean Evidence    : {mean_evidence:.4f}")
print(f"Mean Uncertainty : {mean_uncertainty:.4f}")
