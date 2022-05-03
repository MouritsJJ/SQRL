import argparse
from numpy import Infinity

from utilities import *
from constants import *
from networks import *

def get_model(string):
    if string == 'sqrl':
        return sqrl(num_classes=num_classes, init_weights=init_weights)
    elif string == 'polar':
        return polar(num_classes=num_classes, init_weights=init_weights)
    else:
        print(f'Model {string} not found. Exiting...')
        exit()

def train(model, device, criterion, optimizer, lr_scheduler, data_training, data_validation):
    model.train()
    epoch, best_epoch, best_loss, best_acc = 0, 0, Infinity, 0
    best_model = get_model(args['model'])
    best_model.load_state_dict(model.state_dict())
    while(epoch - best_epoch <= train_patience):
        total_loss = 0
        for i, (images, labels) in enumerate(data_training):
            images, labels = images.to(device), labels.to(device)
            result = model(images)

            loss = criterion(result, labels)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('[Epoch {}, Iteration {} / {}] Avg Loss: {:.6f}'.format(epoch + 1, i + 1, len(data_training), total_loss / (i + 1)), end='\r', flush=True)

        print(f'Epoch {epoch + 1}: ', end='')
        validation_loss, validation_acc = validation(model, device, criterion, data_validation)
        lr_scheduler.step(validation_loss)

        if validation_loss < best_loss:
            best_loss = validation_loss
            best_acc = validation_acc
            print(f'New best loss: {best_loss}')
            best_epoch = epoch
            best_model.load_state_dict(model.state_dict())
        epoch += 1
    model.load_state_dict(best_model.state_dict())
    return best_acc

def main(model_name):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data_training = load_data(p_train)
    data_validation = load_data(p_test)
    data_testing_45 = load_data(p_test_45)
    data_testing_90 = load_data(p_test_90)
    data_testing_360 = load_data(p_test_360)
   
    model = get_model(model_name)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=factor, patience=patience, cooldown=cooldown, min_lr=min_lr)
    validation_acc = train(model, device, criterion, optimizer, lr_scheduler, data_training, data_validation)
    test_loss_45, test_acc_45 = validation(model, device, criterion, data_testing_45, False)
    test_loss_90, test_acc_90 = validation(model, device, criterion, data_testing_90, False)
    test_loss_360, test_acc_360 = validation(model, device, criterion, data_testing_360, False)
    print(f'Test45 loss: {test_loss_45:.6f}, Test acc: {test_acc_45:.3f}')
    print(f'Test90 loss: {test_loss_90:.6f}, Test45 acc: {test_acc_90:.3f}')
    print(f'Test360 loss: {test_loss_360:.6f}, Test90 acc: {test_acc_360:.3f}')
    return validation_acc, test_acc_45, test_acc_90, test_acc_360

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-m','--model', type=str, help='Choose model to train', required=True)
    parser.add_argument('-i','--iter', type=int, help='Iterations', required=True)
    parser.add_argument('-d','--dset', type=str, help='dataset', required=True)
    args = vars(parser.parse_args())

    if args['dset'] not in dataset_paths:
        print('Invalid dataset')
        exit()
    p_train, p_test, p_test_45, p_test_90, p_test_360 = dataset_paths[args['dset']]

    f = open(f"{args['model']}.csv", 'w')
    f.write(f'Validation;-45,45;-90,90;0,360\n')

    for i in range(args['iter']):
        print(f"Test {i+1} / {args['iter']}:")
        x = np.array(main(args['model']))
        f.write(f'{x[0]:.6f};{x[1]:.6f};{x[2]:.6f};{x[3]:.6f}\n')