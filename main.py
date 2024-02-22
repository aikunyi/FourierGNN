import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
from tabulate import tabulate

from data.data_loader import Dataset_ECG, Dataset_Dhfm, Dataset_Solar, Dataset_Wiki,Dataset_OFFWRIST
from model.FourierGNN import FGN
import time
import os
import numpy as np
from utils.utils import save_model, load_model, evaluate
from sklearn import metrics
from torch.nn import functional as F

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# main settings can be seen in markdown file (README.md)
parser = argparse.ArgumentParser(description='fourier graph network for multivariate time series forecasting')
parser.add_argument('--data', type=str, default='offwrist', help='data set')
parser.add_argument('--feature_size', type=int, default='140', help='feature size')
parser.add_argument('--seq_length', type=int, default=12, help='input length')
parser.add_argument('--pre_length', type=int, default=12, help='predict length')
parser.add_argument('--embed_size', type=int, default=128, help='hidden dimensions')
parser.add_argument('--hidden_size', type=int, default=256, help='hidden dimensions')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='input data batch size')
parser.add_argument('--learning_rate', type=float, default=0.00001, help='optimizer learning rate')
parser.add_argument('--exponential_decay_step', type=int, default=5)
parser.add_argument('--validate_freq', type=int, default=1)
parser.add_argument('--early_stop', type=bool, default=False)
parser.add_argument('--decay_rate', type=float, default=0.5)
parser.add_argument('--train_ratio', type=float, default=0.7)
parser.add_argument('--val_ratio', type=float, default=0.3)
parser.add_argument('--device', type=str, default='cuda:0', help='device')

args = parser.parse_args()
print(f'Training configs: {args}')

torch.cuda.empty_cache()

# create output dir
result_train_file = os.path.join('output', args.data, 'train')
result_test_file = os.path.join('output', args.data, 'test')
if not os.path.exists(result_train_file):
    os.makedirs(result_train_file)
if not os.path.exists(result_test_file):
    os.makedirs(result_test_file)

# data set
data_parser = {
    'traffic':{'root_path':'data/traffic.npy', 'type':'0'},
    'ECG':{'root_path':'data/ECG_data.csv', 'type':'1'},
    'COVID':{'root_path':'data/covid.csv', 'type':'1'},
    'electricity':{'root_path':'data/electricity.csv', 'type':'1'},
    'solar':{'root_path':'/data/solar', 'type':'1'},
    'metr':{'root_path':'data/metr.csv', 'type':'1'},
    'wiki':{'root_path':'data/wiki.csv', 'type':'1'},
    'offwrist':{'root_path':"C:/Users/Yuki/OneDrive - The Pennsylvania State University/Documents - WFHN actigraphy/DATA files/4. Dynamic Features/LEEF_combined", 'type':'1'},
}

# data process
if args.data in data_parser.keys():
    data_info = data_parser[args.data]

data_dict = {
    'ECG': Dataset_ECG,
    'COVID': Dataset_ECG,
    'traffic': Dataset_Dhfm,
    'solar': Dataset_Solar,
    'wiki': Dataset_Wiki,
    'electricity': Dataset_ECG,
    'metr': Dataset_ECG,
    'offwrist': Dataset_OFFWRIST,
}

Data = data_dict[args.data]
# train val test
train_set = Data(root_path=data_info['root_path'], flag='train', seq_len=args.seq_length, pre_len=args.pre_length, type=data_info['type'], train_ratio=args.train_ratio, val_ratio=args.val_ratio)
#test_set = Data(root_path=data_info['root_path'], flag='test', seq_len=args.seq_length, pre_len=args.pre_length, type=data_info['type'], train_ratio=args.train_ratio, val_ratio=args.val_ratio)
val_set = Data(root_path=data_info['root_path'], flag='val', seq_len=args.seq_length, pre_len=args.pre_length, type=data_info['type'], train_ratio=args.train_ratio, val_ratio=args.val_ratio)

train_dataloader = DataLoader(
    train_set,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=0,
    drop_last=False,
    pin_memory=True,
)

# test_dataloader = DataLoader(
#     test_set,
#     batch_size=args.batch_size,
#     shuffle=False,
#     num_workers=0,
#     drop_last=False,
#     pin_memory=True,
# )

val_dataloader = DataLoader(
    val_set,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=0,
    drop_last=False,
    pin_memory=True,
)

class NoFussCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    pytorch's CrossEntropyLoss is fussy: 1) needs Long (int64) targets only, and 2) only 1D.
    This function satisfies these requirements
    """

    def forward(self, inp, target):
        return F.cross_entropy(inp, target.long(), weight=torch.tensor([1.0, 10.0, 10.0]).to('cuda'),
                               reduction=self.reduction)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = FGN(pre_length=args.pre_length, embed_size=args.embed_size, feature_size=args.feature_size, seq_length=args.seq_length, hidden_size=args.hidden_size)
my_optim = torch.optim.RMSprop(params=model.parameters(), lr=args.learning_rate, eps=1e-08)
my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=my_optim, gamma=args.decay_rate)
forecast_loss = NoFussCrossEntropyLoss(reduction='none').to(device)
# forecast_loss = nn.CrossEntropyLoss(reduction='none').to(device)
#forecast_loss = nn.MSELoss(reduction='mean').to(device)

def validate(model, vali_loader):
    model.eval()
    cnt = 0
    loss_total = 0
    preds = []
    trues = []
    for i, (x, y) in enumerate(vali_loader):
        cnt += 1
        y = y.to("cuda").squeeze(-1)
        x = x.to("cuda")
        forecast = model(x)

        loss = forecast_loss(forecast, y)
        batch_loss = torch.sum(loss)
        loss = batch_loss/len(loss)
        loss_total += float(loss)
        forecast = forecast.detach().cpu().numpy()
        
        y = y.detach().cpu().numpy()
        preds.append(forecast)
        trues.append(y)


    preds = np.concatenate(preds,axis=0)
    trues = np.concatenate(trues,axis=0).flatten()
    #score = evaluate(trues, preds)

    probs = torch.nn.functional.softmax(torch.from_numpy(preds),dim=1)
    predictions = torch.argmax(probs, dim=1).cpu().numpy()

    print(trues.shape)
    print(predictions.shape)

    
    ConfMatrix = metrics.confusion_matrix(trues, predictions)
    
    label_strings = [0,1,2]
    title='Confusion matrix'
    print(title)
    print(len(title) * '-')
    # Make printable matrix:
    print_mat = []
    for i, row in enumerate(ConfMatrix):
        print_mat.append([label_strings[i]] + list(row))
    print(tabulate(print_mat, headers=['True\Pred'] + label_strings, tablefmt='orgtbl'))

    #model.train()
    total_accuracy = np.trace(ConfMatrix) / len(trues)
    precision, recall, f1, support = metrics.precision_recall_fscore_support(trues, predictions,labels=[0,1,2])
    print(
        {"total_accuracy": total_accuracy, "precision": precision, "recall": recall,
                "f1": f1, "support": support}
    )
    return loss_total/cnt

# def test():
#     result_test_file = 'output/'+args.data+'/train'
#     model = load_model(result_test_file, 48)
#     model.eval()
#     preds = []
#     trues = []
#     sne = []
#     for index, (x, y) in enumerate(test_dataloader):
#         y = y.float().to("cuda:0")
#         x = x.float().to("cuda:0")
#         forecast = model(x)
#         y = y.permute(0, 2, 1).contiguous()
#         forecast = forecast.detach().cpu().numpy()  # .squeeze()
#         y = y.detach().cpu().numpy()  # .squeeze()
#         preds.append(forecast)
#         trues.append(y)

#     preds = np.array(preds)
#     trues = np.array(trues)
#     preds = np.concatenate(preds, axis=0)
#     trues = np.concatenate(trues, axis=0)
#     score = evaluate(trues, preds)
#     print(f'RAW : MAPE {score[0]:7.9%}; MAE {score[1]:7.9f}; RMSE {score[2]:7.9f}.')

def stat_cuda(msg):
    print('--', msg)
    print('allocated: %dM, max allocated: %dM, cached: %dM, max cached: %dM' % (
        torch.cuda.memory_allocated() / 1024 / 1024,
        torch.cuda.max_memory_allocated() / 1024 / 1024,
        torch.cuda.memory_cached() / 1024 / 1024,
        torch.cuda.max_memory_cached() / 1024 / 1024
    ))

if __name__ == '__main__':

    for epoch in range(args.train_epochs):
        epoch_start_time = time.time()
        model.train()
        loss_total = 0
        cnt = 0
        for index, (x, y) in enumerate(train_dataloader):
            
            cnt += 1
            y = y.to("cuda").squeeze(-1)
            x = x.to("cuda")
            
            # print('----------------------------')
            # print(y.min(),y.max())

            forecast = model(x)

            # print(forecast.shape, forecast.dtype)  # Should be [batch_size, num_classes] and torch.float32 (or similar)
            # print(y.shape, y.dtype)  # Should be [batch_size] and torch.long
            # print(forecast)
            
            # print(y.min(),y.max())
            # print('----------------------------')
            
            loss = forecast_loss(forecast, y)

            batch_loss = torch.sum(loss)
            loss = batch_loss/len(loss)
            
            # stat_cuda('compute loss')
            loss.backward()
            # stat_cuda('backward')
            my_optim.step()
            # stat_cuda('optimize')
            loss_total += float(loss)

        if (epoch + 1) % args.exponential_decay_step == 0:
            my_lr_scheduler.step()
        if (epoch + 1) % args.validate_freq == 0:
            val_loss = validate(model, val_dataloader)

        print('| end of epoch {:3d} | time: {:5.2f}s | train_total_loss {:5.4f} | val_loss {:5.4f}'.format(
                epoch, (time.time() - epoch_start_time), loss_total / cnt, val_loss))
        save_model(model, result_train_file, epoch)
