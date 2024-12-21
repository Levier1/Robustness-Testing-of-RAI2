import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import argparse
import pickle
import sys

sys.path.append("../")
import utils
from conf import settings
COPY_NUM = 5

np.random.seed(0)
torch.manual_seed(0)


def sample_intersect1_loader(dataset_name, n_sample=100):
    # Load dataset and select random data points
    # 这里的路径在比对不同情况时记得修改
    (X_vic, y_vic), _ = pickle.load(open(os.path.join(settings.DATA_PATH, 'gaussian_noise', f'{dataset_name.upper()}_intersect_1.0.pkl'), 'rb'))
    np.random.seed(0)
    idx = np.random.choice(len(X_vic), n_sample)
    dataset = utils.SubTrainDataset(X_vic[idx], list(y_vic[idx]), transform=transforms.ToTensor())
    loader = DataLoader(dataset, shuffle=False, batch_size=1)
    return loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', type=str, required=True, help='dataset: cifar10, cifar100, tinyimagenet')
    parser.add_argument('-mc_n_sample', type=int, default=100)
    parser.add_argument('-gpu_id', type=int, default=0, help="GPU device for inference")
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu_id)

    loader = sample_intersect1_loader(args.dataset, args.mc_n_sample)
    verify_samples = []
    for sample, y in loader:
        verify_samples.append(sample)
    verify_tensor = torch.cat(verify_samples)
    print("Initialized verification set.")

    victim_model_type = 'vgg11'
    attacker_model_type = 'vgg19'

    # 这里的路径在比对不同情况时记得修改 'model_type_1'为受害者模型储存路径 'model_type_2'为攻击者模型存储路径
    file_path1 = os.path.join(settings.CHECKPOINT_PATH, 'noise_similarity', args.dataset)
    file_path2 = os.path.join(settings.CHECKPOINT_PATH, 'noise_similarity', args.dataset)

    # 这里记得同步修改
    model_type_list = [victim_model_type, attacker_model_type]  # Specify your two model types
    inter_names = os.listdir(
        os.path.join(file_path1, model_type_list[0]))  # Assuming both model types have the same intersection names
    mean_std_dict = utils.get_intersection_mean_std_dict(args.dataset)

    result_path = os.path.join(settings.RESULT_PATH, "dataset_similarity_noise_tinyimagenet")
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    print(mean_std_dict.keys())
    print(inter_names)

    # Load models into model_dict
    all_model_dict = {}
    for model_type in model_type_list:
        print(model_type)

        model_predict_on_neighrboods = {}
        for intersection in inter_names:
            mean, std = mean_std_dict[intersection]
            models_list = []
            for i in range(COPY_NUM):
                # 同样需要修改'model_type_1'变量
                if model_type == victim_model_type:
                    model_path = os.path.join(file_path1, model_type_list[0], intersection, 'model_{}.pth'.format(i))
                else:
                    model_path = os.path.join(file_path2, model_type_list[1],intersection, 'model_{}.pth'.format(i))

                net = utils.get_network(args.dataset, model_type, False)
                net.load_state_dict(torch.load(model_path, map_location='cpu'))
                net.to('cpu')
                net.eval()
                models_list.append(nn.Sequential(transforms.Normalize(mean, std), net))
            print("Initialized model {}/{}.".format(model_type, intersection))

            with torch.no_grad():
                model_predicts = []
                for model in models_list:
                    model.cuda()
                    model_predicts.append(model(verify_tensor.cuda()).softmax(dim=1))
                    del model
                    torch.cuda.empty_cache()
                model_predict_on_neighrboods[intersection] = torch.stack(model_predicts).detach().cpu()

        pickle.dump(model_predict_on_neighrboods,
                    open(os.path.join(result_path, f"{args.dataset}_predict_{model_type}.pkl"), "wb"))
