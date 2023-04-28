import pickle
import os
import pandas as pd
from tqdm import tqdm
from src.models import *
from src.constants import *
from src.plotting import *
from src.pot import *
from src.utils import *
from src.diagnosis import *
from src.merlin import *
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
from time import time
from pprint import pprint
# from beepy import beep
import pywt
from scipy.fftpack import fft, ifft
import heapq
global ffnn
ffnn = ""
def fft_deal(data, small_rate=0.3):
    filtered_data = []
    min_list = []
    for i in range(data.shape[1]):
        fft_data = fft(data[:, i])
        r_fft_data = fft_data
        r_fft_data_l = []
        for j in range(len(r_fft_data)):
            r_fft_data_l.append(np.abs(r_fft_data[j]))

        min_number = heapq.nsmallest(int(len(r_fft_data_l) * small_rate), r_fft_data_l)
        min_index = []
        for t in min_number:
            index = r_fft_data_l.index(t)
            min_index.append(index)
        fft_data[min_index] = 0
        fft_data = ifft(fft_data)
        fft_data = fft_data.real
        filtered_data.append(fft_data)
    filtered_data = np.array(filtered_data)
    filtered_data = filtered_data.transpose()
    f = filtered_data.real
    return f


def filter_deal(data, wavename, name, threshold, if_output_picture=False):
    if wavename == '':
        return data
    if if_output_picture:
        plt.figure()
        plt.plot(data, linestyle="-", linewidth=0.1)
        plt.title(name + "1")
    filtered_data = []

    for i in range(data.shape[1]):
        sum = 0
        for j in data[:, i]:
            sum += j * j
        if sum == 0:
            apd = data[:, i]

            filtered_data.append(apd)
            continue


        w = pywt.Wavelet(wavename)  
        maxlev = pywt.dwt_max_level(len(data[:, i]), w.dec_len)

        coeffs = pywt.wavedec(data[:, i], wavename, level=maxlev)  

        for i in range(1, len(coeffs)):
            coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))  

        apd = pywt.waverec(coeffs, wavename)  
        for i in range(data.shape[1]):
            if apd.shape[0] - 1 == data[:, i].shape[0]:
                apd = apd[range(0, data[:, i].shape[0])]

        filtered_data.append(apd)
    filtered_data = np.array(filtered_data)
    filtered_data = pd.DataFrame(filtered_data)
    filtered_data = filtered_data.fillna(1)
    filtered_data = filtered_data.values
    filtered_data = filtered_data.transpose()
    filtered_data = filtered_data / filtered_data.max()
    if if_output_picture:
        plt.figure()
        plt.plot(filtered_data, linestyle="-", linewidth=0.1)
        plt.title(name + "2")
        plt.show()
    return filtered_data


def convert_to_windows(data, model):
    windows = [];
    w_size = model.n_window
    for i, g in enumerate(data):
        if i >= w_size:
            w = data[i - w_size:i]
        else:
            w = torch.cat([data[0].repeat(w_size - i, 1), data[0:i]])
        windows.append(w if 'TranAD' in args.model or 'Attention' in args.model else w.view(-1))
    window_stack = torch.stack(windows)
      return window_stack


def load_dataset(dataset, data_name):
    global ffnn
    folder = os.path.join(output_folder, dataset)
    print("folder", folder)
    if not os.path.exists(folder):
        raise Exception('Processed Data not found.')
    loader = []
    for file in ['train', 'test', 'labels']:
        if dataset == 'SMD': file = args.subset + '_' + file
        if dataset == 'SMAP': file = args.subset + '_' + file
        if dataset == 'MSL':
            file = args.subset + '_' + file
            print("file", file)
        if dataset == 'UCR': file = args.subset + '_' + file
        if dataset == 'NAB': file = 'ec2_request_latency_system_failure_' + file
        ffnn = file
        loader.append(np.load(os.path.join(folder, f'{file}.npy')))
    
    tl1_train = loader[0]
    tl1_test = loader[1]
        tl2_train = fft_deal(loader[0])
    tl2_test = fft_deal(loader[1])
    tl3_train = filter_deal(loader[0], filter3_train, "train", f3_train_par)
    tl3_test = filter_deal(loader[1], filter3_test, "test", f3_test_par)

    if args.less: loader[0] = cut_array(0.2, loader[0])
    train_loader1 = DataLoader(tl1_train, batch_size=loader[0].shape[0])
    test_loader1 = DataLoader(tl1_test, batch_size=loader[1].shape[0])
    train_loader2 = DataLoader(tl2_train, batch_size=loader[0].shape[0])
    test_loader2 = DataLoader(tl2_test, batch_size=loader[1].shape[0])
    train_loader3 = DataLoader(tl3_train, batch_size=loader[0].shape[0])
    test_loader3 = DataLoader(tl3_test, batch_size=loader[1].shape[0])
    labels = loader[2]
    return train_loader1, test_loader1, train_loader2, test_loader2, train_loader3, test_loader3, labels


def save_model(model, optimizer, scheduler, epoch, accuracy_list):
    folder = f'checkpoints/{args.model}_{args.dataset}/'
    os.makedirs(folder, exist_ok=True)
    file_path = f'{folder}/model.ckpt'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy_list': accuracy_list}, file_path)


def load_model(modelname, dims):
        import src.models

    model_class = getattr(src.models, modelname)
    model = model_class(dims).double()
    optimizer = torch.optim.AdamW(model.parameters(), lr=model.lr, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
    fname = f'checkpoints/{args.model}_{args.dataset}/model.ckpt'

    if os.path.exists(fname) and (not args.retrain or args.test):
        print(f"{color.GREEN}Loading pre-trained model: {model.name}{color.ENDC}")
        checkpoint = torch.load(fname)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        accuracy_list = checkpoint['accuracy_list'
    else:
        print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
        epoch = -1
        accuracy_list = []
    return model, optimizer, scheduler, epoch, accuracy_list



def backprop(epoch, model, data1, dataO1, data2, dataO2, data3, dataO3, optimizer, dataset_name, scheduler,
             training=True, update=False):
    global w1, w2, w3

    l = nn.MSELoss(reduction='mean' if training else 'none')

    feats1 = dataO1.shape[1]
    feats2 = dataO2.shape[1]
    feats3 = dataO3.shape[1]

    print("dataset_name", dataset_name)

    if 'GAN' in model.name:

        print("back-l")
        l = nn.MSELoss(reduction='none')

        bcel = nn.BCELoss(reduction='mean')

        msel = nn.MSELoss(reduction='mean')

        real_label, fake_label = torch.tensor([0.99]), torch.tensor([0.01])

        real_label, fake_label = real_label.type(torch.DoubleTensor), fake_label.type(torch.DoubleTensor)
        n = epoch + 1;
        w_size = model.n_window

        mses, gls, dls = [], [], []


        if training:

            iter_training = 0

            for iiiii in range(data1.shape[0]):
                d1 = data1[iiiii]
                d2 = data2[iiiii]
                d3 = data3[iiiii]
                iter_training += 1
                model.discriminator.zero_grad()

                _, real1, fake1 = model(d1)
                _, real2, fake2 = model(d2)
                _, real3, fake3 = model(d3)

                dl1 = bcel(real1, real_label) + bcel(fake1, fake_label)
                dl2 = bcel(real2, real_label) + bcel(fake2, fake_label)
                dl3 = bcel(real3, real_label) + bcel(fake3, fake_label)
                if update:
                    if dl1 <= dl2 and dl1 <= dl3:
                        w1 = 1
                        w2 = 0
                        w3 = 0

                    elif dl2 <= dl1 and dl2 <= dl3:
                        w1 = 0
                        w2 = 1
                        w3 = 0
                    elif dl3 <= dl1 and dl3 <= dl2:
                        w1 = 0
                        w2 = 0
                        w3 = 1
                dl_av = w1 * dl1 + w2 * dl2 + w3 * dl3

                
                dl_av.backward()
                
                if 'myGAN' in model.name:
                    model.gen.zero_grad()
                    model.rth.zero_grad()
                elif 'MAD_GAN' in model.name:
                    model.generator.zero_grad()

                optimizer.step()

                
                z1, _, fake1 = model(d1)
                z2, _, fake2 = model(d2)
                z3, _, fake3 = model(d3)

                mse_av = w1 * msel(z1, d1) + w2 * msel(z2, d2) + w3 * msel(z3, d3)

                gl_av = w1 * bcel(fake1, real_label) + w2 * bcel(fake2, real_label) + w3 * bcel(fake3, real_label)
               
                tl = gl_av + mse_av
                tl.backward()
                model.discriminator.zero_grad()
                optimizer.step()
                mses.append(mse_av.item())
                gls.append(gl_av.item())
                dls.append(dl_av.item())

            tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(mses)},\tG = {np.mean(gls)},\tD = {np.mean(dls)}')
            return np.mean(gls) + np.mean(dls), optimizer.param_groups[0]['lr']


        else:

            outputs1 = []
            outputs2 = []
            outputs3 = []

            print("data.shape", data1.shape)
            for iiiii in range(data1.shape[0]):
                d1 = data1[iiiii]
                d2 = data2[iiiii]
                d3 = data3[iiiii]
                z1, _, _ = model(d1)
                z2, _, _ = model(d2)
                z3, _, _ = model(d3)
                outputs1.append(z1)
                outputs2.append(z2)
                outputs3.append(z3)

            outputs1 = torch.stack(outputs1)
            outputs2 = torch.stack(outputs2)
            outputs3 = torch.stack(outputs3)

            y_pred1 = outputs1[:, data1.shape[1] - feats1:data1.shape[1]].view(-1, feats1)
            y_pred2 = outputs2[:, data2.shape[1] - feats2:data2.shape[1]].view(-1, feats2)
            y_pred3 = outputs3[:, data3.shape[1] - feats3:data3.shape[1]].view(-1, feats3)


            loss1 = l(outputs1, data1)
            loss2 = l(outputs2, data2)
            loss3 = l(outputs3, data3)

            loss1 = loss1[:, data1.shape[1] - feats1:data1.shape[1]].view(-1, feats1)
            loss2 = loss2[:, data2.shape[1] - feats2:data2.shape[1]].view(-1, feats2)
            loss3 = loss3[:, data3.shape[1] - feats3:data3.shape[1]].view(-1, feats3)
            res1_1 = loss1.detach().numpy()
            res1_2 = loss2.detach().numpy()
            res1_3 = loss3.detach().numpy()
            res2_1 = y_pred1.detach().numpy()
            res2_2 = y_pred2.detach().numpy()
            res2_3 = y_pred3.detach().numpy()
            res1 = w1 * res1_1 + w2 * res1_2 + w3 * res1_3
            res2 = w1 * res2_1 + w2 * res2_2 + w3 * res2_3
            return res1, res2



def not_all_empty_col_in_dataframe(data):
    not_empty_index = []
    i = 0
    for name, col in data.iteritems():
        if not col.isnull().all():
            not_empty_index.append(i)
        i += 1
    data = data.iloc[:, not_empty_index]
    return data


if __name__ == '__main__':
    global w1, w2, w3, file, filter2_train, filter2_test, filter3_train, filter3_test, data_file
    data_name = args.dataset
    print("argsq.", args.q)
    if data_name == 'WADI':
        w1 = 0
        w2 = 0
        w3 = 1
    elif data_name == 'SWaT':
        w1 = 1
        w2 = 0
        w3 = 0
    elif data_name == 'MSDS':
        w1 = 1
        w2 = 0
        w3 = 0
    elif data_name == 'MSL':
        w1 = 1
        w2 = 0
        w3 = 0
    elif data_name == 'SMD':
        w1 = 1
        w2 = 0
        w3 = 0
    elif data_name == 'UCR':
        data_file = '138'
        if data_file == '135':
            w1 = 1
            w2 = 0
            w3 = 0
        if data_file == '136':
            w1 = 1
            w2 = 0
            w3 = 0
        if data_file == '137':
            w1 = 1
            w2 = 0
            w3 = 0
        if data_file == '138':
            w1 = 0
            w2 = 0
            w3 = 1
    filter2_train = ''
    filter2_test = ''
    filter3_train = ''
    filter3_test = ''
    if data_name == 'WADI':
        filter2_train = 'fft'
        f2_train_par = 1
        filter2_test = 'fft'
        f2_test_par = 1
        filter3_train = 'db1'
        f3_train_par = 1
        filter3_test = 'db1'
        f3_test_par = 1
    elif data_name == 'SWaT':
        filter2_train = 'fft'
        f2_train_par = 1
        filter2_test = 'fft'
        f2_test_par = 1
        filter3_train = 'db8'
        f3_train_par = 1
        filter3_test = 'db8'
        f3_test_par = 1
    elif data_name == 'MSL':
        filter2_train = 'fft'
        f2_train_par = 1
        filter2_test = 'fft'
        f2_test_par = 1
        filter3_train = 'db8'
        f3_train_par = 1
        filter3_test = 'db8'
        f3_test_par = 1
    elif data_name == 'MSDS':
        filter2_train = 'fft'
        f2_train_par = 1
        filter2_test = 'fft'
        f2_test_par = 1
        filter3_train = 'db8'
        f3_train_par = 1
        filter3_test = 'db8'
        f3_test_par = 1
    elif data_name == 'UCR':

        filter2_train = 'fft'
        f2_train_par = 1
        filter2_test = 'fft'
        f2_test_par = 1
        filter3_train = 'db8'
        f3_train_par = 1
        filter3_test = 'db8'
        f3_test_par = 1
    elif data_name == 'SMD':
        filter2_train = 'fft'
        f2_train_par = 1
        filter2_test = 'fft'
        f2_test_par = 1
        filter3_train = 'db8'
        f3_train_par = 1
        filter3_test = 'db8'
        f3_test_par = 1


    train_loader1, test_loader1, train_loader2, test_loader2, train_loader3, test_loader3, labels = load_dataset(
        args.dataset, data_name)
    if args.model in ['MERLIN']:
        eval(f'run_{args.model.lower()}(test_loader, labels, args.dataset)')

    model, optimizer, scheduler, epoch, accuracy_list = load_model(args.model, labels.shape[1])


    trainD1, testD1 = next(iter(train_loader1)), next(iter(test_loader1))
    trainD2, testD2 = next(iter(train_loader2)), next(iter(test_loader2))
    trainD3, testD3 = next(iter(train_loader3)), next(iter(test_loader3))

    trainO1, testO1 = trainD1, testD1
    trainO2, testO2 = trainD2, testD2
    trainO3, testO3 = trainD3, testD3
    if model.name in ['Attention', 'DAGMM', 'USAD', 'MSCRED', 'CAE_M', 'GDN', 'MTAD_GAT', 'MAD_GAN',
                      'myGAN'] or 'TranAD' in model.name:
        trainD1, testD1 = convert_to_windows(trainD1, model), convert_to_windows(testD1, model)
        trainD2, testD2 = convert_to_windows(trainD2, model), convert_to_windows(testD2, model)
        trainD3, testD3 = convert_to_windows(trainD3, model), convert_to_windows(testD3, model)


    import datetime

    t1 = datetime.datetime.now()
    if not args.test:

        print(f'{color.HEADER}Training {args.model} on {args.dataset}{color.ENDC}')
        num_epochs = 5;
       
        e = epoch + 1;
        start = time()

        for e in tqdm(list(range(epoch + 1, epoch + num_epochs + 1))):
            
            lossT, lr = backprop(e, model, trainD1, trainO1, trainD2, trainO2, trainD3, trainO3, optimizer, data_name,
                                 scheduler, True)
            print("lossT", lossT, "lr", lr)

            accuracy_list.append((lossT, lr))

        print(color.BOLD + 'Training time: ' + "{:10.4f}".format(time() - start) + ' s' + color.ENDC)

        save_model(model, optimizer, scheduler, e, accuracy_list)


        plot_accuracies(accuracy_list, f'{args.model}_{args.dataset}')
    t2 = datetime.datetime.now()


    torch.zero_grad = True

    model.eval()

    print(f'{color.HEADER}Testing {args.model} on {args.dataset}{color.ENDC}')

    loss, y_pred = backprop(0, model, testD1, testO1, testD2, testO2, testD3, testO3, optimizer, data_name, scheduler,
                            training=False)



    if not args.test:
        plotter(f'{args.model}_{args.dataset}', testO1, y_pred, loss, labels)
        plotter(f'{args.model}_{args.dataset}', testO2, y_pred, loss, labels)
        plotter(f'{args.model}_{args.dataset}', testO3, y_pred, loss, labels)

    df = pd.DataFrame()


    lossT, _ = backprop(0, model, trainD1, trainO1, trainD2, trainO2, trainD3, trainO3, optimizer, data_name, scheduler,
                       training=False)

    iters = 0
    for i in range(loss.shape[1]):
        iters += 1
        lt, l, ls = lossT[:, i], loss[:, i], labels[:, i]
        
        if lt.sum() == 0:
            continue
        result, pred = pot_eval(lt, l, ls,args.q)
        preds.append(pred)
        df = df.append(result, ignore_index=True)
    
    lossTfinal, lossFinal = np.mean(lossT, axis=1), np.mean(loss, axis=1)
    labelsFinal = (np.sum(labels, axis=1) >= 1) + 0
    result, _ = pot_eval(lossTfinal, lossFinal, labelsFinal, args.q)
    result.update(hit_att(loss, labels))
    result.update(ndcg(loss, labels))
    print("df**", df)
    print("result**", " file:",ffnn, " ",result)
    t3 = datetime.datetime.now()
    print("training time:", t2 - t1)
    print("testing time:", t3 - t2)
    print(args.q)

    s = str(args.dataset) + " " + str(args.subset) + " q=" + str(args.q) + " ws=" + str(model.n_window) + " mask=" + str(model.parameters)
    s2 = str(result)
    s3 = str("training time:" + str(t2 - t1) + " testing time:" + str(t3 - t2))
    s4 = str(args.q)
    f = open(args.exefile, 'a')
    f.write(s)
    f.write("\n")
    f.write(s2)
    f.write("\n")
    f.write(s3)
    f.write("\n")
    f.write(s4)
    f.write("\n\n")
    f.close()
