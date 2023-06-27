from CIFAR_Dataset import cifar_dataset
from CIFAR_Dataset import symmetrical_noisy, asymmetrical_cifar10, asymmetrical_cifar100
import torch.backends.cudnn as cudnn
import os
import argparse
import random
import torch
import torchvision.transforms as transforms
from model import *
import numpy as np
import torch.nn.functional as F
from opt import k_barycenter, label_reg, label_propagation_analysis
import torch.optim as optim
from scipy.spatial.distance import cdist
from data_augment import CIFAR10Policy

parser = argparse.ArgumentParser(description='Noisy CIFAR Training')
parser.add_argument('--batch_size', default=128, type=int) 
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float)
parser.add_argument('--rampup_length', default=0, type=int)
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--seed', default=2023)
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--alpha', default=4, type=float)
parser.add_argument('--lambda_u', default=25, type=float)
parser.add_argument('--lambd', default=10, type=float)
parser.add_argument('--T', default=0.5, type=float)
parser.add_argument('--barycenter_number', default=1, type=int)
parser.add_argument('--noisy_rate', default=0.5, type=float)
parser.add_argument('--noisy_type', default='sym', type=str)
parser.add_argument('--device', default='3', type=str)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.device
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if args.dataset=='cifar10':
    root = './data/cifar-10-batches-py'
    num_class = 10
    warm_up = 10
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_strong_10 = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        CIFAR10Policy(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
        
    labeled_transform = [transform_strong_10, transform_strong_10]
    unlabeled_transform = [transform_strong_10, transform_strong_10, transform_train, transform_train]

else:
    root = './data/cifar-100-python'
    num_class = 100
    warm_up = 30
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)),
    ])
    
    transform_strong_100 = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        CIFAR10Policy(),
        transforms.ToTensor(),
        transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
    ])
    
    labeled_transform = [transform_strong_100, transform_strong_100]
    unlabeled_transform = [transform_strong_100, transform_strong_100, transform_train, transform_train]

CIFAR_test = cifar_dataset(args.dataset, transform_test, 'test', root, selected = [], noisy_label = [])
CIFAR_clean_train = cifar_dataset(args.dataset, transform_train, 'train', root, selected = [], noisy_label = [])

# generate noisy label
if args.noisy_type=='sym':
    noisy_label = symmetrical_noisy(CIFAR_clean_train.train_label, args.noisy_rate, num_class)
elif args.noisy_type=='asym' and args.dataset=='cifar10':
    noisy_label = asymmetrical_cifar10(CIFAR_clean_train.train_label, args.noisy_rate)
elif args.noisy_type=='asym' and args.dataset=='cifar100':
    noisy_label = asymmetrical_cifar100(CIFAR_clean_train.train_label, args.noisy_rate)

CIFAR_noisy_train = cifar_dataset(args.dataset, transform_train, 'train', root, selected = [], noisy_label = noisy_label)

train_loader = torch.utils.data.DataLoader(CIFAR_noisy_train, args.batch_size, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(CIFAR_test, args.batch_size, shuffle=False, num_workers=2)

def create_model():
    model = ResNet18(num_classes = num_class)
    model = model.cuda()
    return model

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, args.lambda_u * linear_rampup(epoch)

def linear_rampup(current, rampup_length=args.rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)    

def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]    

def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets

def adjust_learning_rate(optimizer, epoch):
    if epoch==150:
        args.lr = args.lr * 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr

def train_CE(train_loader, model, optimizer,epoch):
    for batch_idx, (inputs, targets, sample_id) in enumerate(train_loader):
        model.train()
        inputs, targets = inputs.cuda(), targets.cuda()        
        optimizer.zero_grad()
        outputs, feature = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()

def test(test_loader, model, epoch):
    test_loss = 0
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs, feature = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100.*correct/total
    print("test accuracy:",acc)
    return acc

net = create_model()
cudnn.benchmark = True
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
train_criterion = SemiLoss()

def MixMatch(labeled_trainloader, unlabeled_trainloader, model, optimizer, criterion, epoch):    
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)

    model.train()
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)
    for batch_idx in range(num_iter):
        try:
            inputs_x, inputs_x2, labels_x = next(labeled_train_iter)
        
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, inputs_x2, labels_x = next(labeled_train_iter)

        try:
            inputs_u, inputs_u2, inputs_u3, inputs_u4 = next(unlabeled_train_iter)
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2, inputs_u3, inputs_u4 = next(unlabeled_train_iter)


        batch_size = inputs_x.size(0)
        labels_x = torch.zeros(batch_size, num_class).scatter_(1, labels_x.view(-1,1).long(), 1)        
        inputs_x, inputs_x2, labels_x = (inputs_x.cuda(), inputs_x2.cuda(), labels_x.cuda())        
        inputs_u, inputs_u2, inputs_u3, inputs_u4 = (inputs_u.cuda(), inputs_u2.cuda(), inputs_u3.cuda(), inputs_u4.cuda())

        with torch.no_grad():
            outputs_u_1, feature = model(inputs_u3)
            outputs_u_2, feature = model(inputs_u4)                       
            p = (torch.softmax(outputs_u_1, dim=1) + torch.softmax(outputs_u_2, dim=1)) / 2
            pt = p**(1/args.T)
            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()

        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([labels_x, labels_x, targets_u, targets_u], dim=0)
        l = np.random.beta(args.alpha, args.alpha)
        l = max(l, 1-l)
        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b
       
        logits, feature = model(mixed_input)
        logits_x = logits[:batch_size*2]
        logits_u = logits[batch_size*2:]

        Lx, Lu, w = criterion(logits_x, mixed_target[:batch_size*2], logits_u, mixed_target[batch_size*2:], epoch+batch_idx/num_iter)

        loss = Lx + w * Lu
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
def OT_Filter(train_loader, model, optimizer, epoch):
    model.eval()
    train_feature = []
    train_label = np.array([])
    train_sample_id = np.array([])
    
    for batch_idx, (inputs, targets, sample_id) in enumerate(train_loader):
        label = targets.numpy()
        sample_id = sample_id.numpy()
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs, feature = model(inputs)
        feature = feature.detach().cpu().numpy()       
        feature = feature.tolist()
        
        train_feature+=feature
        train_label = np.concatenate((train_label, label),axis = 0)
        train_sample_id = np.concatenate((train_sample_id, sample_id),axis = 0)
    train_feature = np.array(train_feature)
    
    barycenter = []
    barycenter_class = []
    for class_id in range(num_class):
        sample = np.where(train_label==class_id)
        sample_feature = train_feature[sample]
        center = k_barycenter(sample_feature.transpose(),args.barycenter_number, args.lambd)
        barycenter_class+=[class_id for i in range(args.barycenter_number)]
        barycenter+=center.transpose().tolist()
    
    barycenter = np.array(barycenter)
    barycenter_weight = np.ones(args.barycenter_number*num_class)/(args.barycenter_number*num_class)
    train_sample_weight = np.ones(len(train_feature))/len(train_feature)
    
    cost_matrix = cdist(barycenter, train_feature, metric='euclidean')       
    opt_result = label_reg(barycenter_weight, train_sample_weight, cost_matrix, barycenter_class, num_class, args.lambd)
    probability, label_propagation = label_propagation_analysis(opt_result, barycenter_class, num_class)
    
    # get clean barycenter
    EM_feature = train_feature[np.where(train_label==label_propagation)]
    EM_label = train_label[np.where(train_label==label_propagation)]
    EM_barycenter = []
    barycenter_class = []
    for class_id in range(num_class):
        sample = np.where(EM_label==class_id)
        sample_feature = EM_feature[sample]
        center = k_barycenter(sample_feature.transpose(),args.barycenter_number, args.lambd)
        barycenter_class+=[class_id for i in range(args.barycenter_number)]
        EM_barycenter+=center.transpose().tolist()
    
    EM_barycenter = np.array(EM_barycenter)
            
    cost_matrix = cdist(EM_barycenter, train_feature, metric='euclidean')       
    opt_result = label_reg(barycenter_weight, train_sample_weight, cost_matrix, barycenter_class, num_class, args.lambd)
    probability, label_propagation = label_propagation_analysis(opt_result, barycenter_class, num_class)
        
    labeled_sample = train_sample_id[np.where(train_label==label_propagation)]
    unlabeled_sample = train_sample_id[np.where(train_label!=label_propagation)]
    
    labeled_dataset = cifar_dataset(args.dataset, labeled_transform, 'labeled', root, selected = labeled_sample, noisy_label = noisy_label)
    unlabeled_dataset = cifar_dataset(args.dataset, unlabeled_transform, 'unlabeled', root, selected = unlabeled_sample, noisy_label = noisy_label)
    
    labeled_trainloader = torch.utils.data.DataLoader(
    labeled_dataset, args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    
    unlabeled_trainloader = torch.utils.data.DataLoader(
    unlabeled_dataset, args.batch_size, shuffle=True, num_workers=0, drop_last=True)    
    MixMatch(labeled_trainloader, unlabeled_trainloader, model, optimizer, train_criterion, epoch)

if __name__ == '__main__':
    for epoch in range(args.num_epochs):
        print("epoch",epoch)
        adjust_learning_rate(optimizer, epoch)
        
        if epoch<warm_up:
            train_CE(train_loader, net, optimizer,epoch)
            test(test_loader, net, epoch)
       
        else:
            OT_Filter(train_loader, net, optimizer, epoch)
            test(test_loader, net, epoch)

    