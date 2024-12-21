import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import argparse
import pickle
import utils
import utils_noise # 记得修改变量
from conf import settings
from utils_noise import get_intersection_mean_std  # 记得修改变量

def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1, 5)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, y_pred = output.topk(k=maxk, dim=1)
        y_pred = y_pred.t()

        target_reshaped = target.view(1, -1).expand_as(y_pred)
        correct = (y_pred == target_reshaped)

        list_topk_accs = []
        for k in topk:
            ind_which_topk_matched_truth = correct[:k]
            flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(-1).float()
            tot_correct_topk = flattened_indicator_which_topk_matched_truth.sum(dim=0, keepdim=True)
            topk_acc = tot_correct_topk / batch_size
            list_topk_accs.append(topk_acc.item())
        return list_topk_accs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', type=str, required=True, help='dataset: cifar10, cifar100, tinyimagenet')
    parser.add_argument('-gpu_id', type=int, default=0, help='gpu id used for inference')
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu_id)

    test_loader = utils_noise.get_test_dataloader(args.dataset, mean=(0, 0, 0), std=(1, 1, 1), batch_size=512, num_workers=8, shuffle=False, pin_memory=False)

    # file_path = os.path.join(settings.CHECKPOINT_PATH, 'similarity', args.dataset)
    result_file = "/home/featurize/work/RAI2project/RAI2result/RotationTest_accuracy-1.csv"

    # 写入CSV标题行
    with open(result_file, "a") as f:
        f.write("Dataset,Intersection,Model,Top-1,Top-5\n")

    # 只评估特定模型架构
    victim_model_type = 'resnet34'
    attacker_model_type = 'resnet101'

    # 遍历 inter_propor 从 0.0 到 1.0
    for inter_propor in np.arange(0.0,1.1,0.1):
        inter_propor = round(inter_propor, 1)
        inter_propor_str = f"int{inter_propor}"

        # 获取数据标准化的 mean 和 std
        mean, std = get_intersection_mean_std(args.dataset, inter_propor)

        # 受害者模型
        victim_model_path = os.path.join(settings.CHECKPOINT_PATH_SHEAR, 'similarity', args.dataset, victim_model_type, inter_propor_str, 'model_0.pth')
        victim_net = utils.get_network(args.dataset, victim_model_type, False)
        victim_net.load_state_dict(torch.load(victim_model_path, map_location='cpu'))
        victim_net.eval()

        # 攻击者模型
        attacker_model_path = os.path.join(settings.CHECKPOINT_PATH_SHEAR, 'similarity', args.dataset, attacker_model_type, inter_propor_str, 'model_2.pth')
        attacker_net = utils.get_network(args.dataset, attacker_model_type, False)
        attacker_net.load_state_dict(torch.load(attacker_model_path, map_location='cpu'))
        attacker_net.eval()

        print(f"Evaluating victim model: {victim_model_type} and attacker model: {attacker_model_type} on intersection proportion {inter_propor_str}...")

        # 包装模型以便在推断时应用标准化
        victim_model = nn.Sequential(transforms.Normalize(mean, std), victim_net)
        attacker_model = nn.Sequential(transforms.Normalize(mean, std), attacker_net)

        # 在测试集上进行评估
        with torch.no_grad():
            # 评估受害者模型
            victim_outputs = []
            targets = []
            for x, y in test_loader:
                victim_outputs.append(victim_model(x))
                targets.append(y.reshape(-1, 1))

            victim_outputs = torch.cat(victim_outputs, dim=0)
            targets = torch.cat(targets, dim=0)
            top1_acc, top5_acc = accuracy(victim_outputs, targets, topk=(1, 5))

            # 写入受害者模型结果到CSV
            with open(result_file, "a") as f:
                f.write(f"{args.dataset},{inter_propor},Victim ({victim_model_type}),{top1_acc:.5f},{top5_acc:.5f}\n")
            print(f"Dataset: {args.dataset}, Intersection: {inter_propor}, Victim ({victim_model_type}), {top1_acc:.5f}, {top5_acc:.5f}\n")

            # 评估攻击者模型
            attacker_outputs = []
            for x, y in test_loader:
                attacker_outputs.append(attacker_model(x))

            attacker_outputs = torch.cat(attacker_outputs, dim=0)
            top1_acc, top5_acc = accuracy(attacker_outputs, targets, topk=(1, 5))

            # 写入攻击者模型结果到CSV
            with open(result_file, "a") as f:
                f.write(f"{args.dataset},{inter_propor},Attacker ({attacker_model_type}),{top1_acc:.5f},{top5_acc:.5f}\n")
            print(f"Dataset: {args.dataset}, Intersection: {inter_propor}, Attacker ({attacker_model_type}), {top1_acc:.5f}, {top5_acc:.5f}\n")

    print(f"Accuracy results saved to {result_file}")
