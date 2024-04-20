import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch import nn
import torch.nn.functional as F


def main(train_model=False):
    ############################################################
    #    Data Preprocessing
    ############################################################

    df = pd.read_csv("./train.csv", index_col='id')

    sex_dummies = pd.get_dummies(df['Sex'], prefix='Sex')
    df = pd.concat([df, sex_dummies], axis=1)
    df.drop(['Sex'], axis=1, inplace=True)

    y_torch = torch.tensor(df['Rings'].values, dtype=torch.int32)
    df.drop(['Rings'], axis=1, inplace=True)
    X_torch = torch.tensor(df.values.astype(np.float32)).unsqueeze(1)

    dataset = TensorDataset(X_torch, y_torch)


    ############################################################
    #   Train split
    ############################################################

    train_size = int(0.90 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


    batch_size = 2
    num_workers = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


    ############################################################
    #   Neural Net Model in PyTorch
    ############################################################

    num_features = X_torch.shape[2]
    print(f'Features {num_features} with X train size of {train_size}')

    class Regressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear( in_features = num_features, out_features = 512)
            self.dropout1 = nn.Dropout(0.25) # 25% dropout
            self.fc2 = nn.Linear( in_features = 512, out_features = 256)
            self.fc3 = nn.Linear( in_features = 256, out_features = 128)
            self.fc4 = nn.Linear( in_features = 128, out_features = 1)

        def forward(self, x):
            x = F.relu( self.fc1(x))
            x = self.dropout1(x)
            x = F.relu( self.fc2(x))
            x = F.relu( self.fc3(x))
            x = self.fc4(x)
            # x = torch.relu(x) # apply ReLu to ensure the output is non-negative
            return x


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    model = Regressor().to(device)


    ############################################################
    #   Loss Function:  Root Mean Squared Logrithmic Error
    ############################################################
    ''' numpy version
    def rmsle(y_true, y_pred):
        log_diff = np.log1p(y_pred) - np.log1p(y_true)
        return np.sqrt(np.mean(log_diff**2))
    '''
    # torch version
    def rmsle(y_true, y_pred):
        # Ensure the predictions are positive since log1p requires positive numbers
        # y_pred = torch.round(y_pred)
        # y_pred = torch.clamp(y_pred, min=0)

        log_true = torch.log1p(y_true)
        log_pred = torch.log1p(y_pred)
        log_diff = log_pred - log_true
        return torch.sqrt(torch.mean(log_diff ** 2))

    ############################################################
    #   Model Training
    ############################################################

    epochs = 40
    lr = 0.01

    opt = torch.optim.Adam(model.parameters(), lr)

    val_loss_min = np.inf
    stop_cnt = 0
    stop_criteria = 8

    if train_model:
        for ep in range(epochs):

            model.train()
            for X_test, y_test in train_loader:

                X_batch, y_test = X_test.to(device), y_test.to(device).unsqueeze(1)

                opt.zero_grad()

                y_pred = model(X_batch)
                loss = rmsle(y_test, y_pred)

                loss.backward()
                opt.step()


            print(f'Epoch: {ep}, Loss: {loss.item()}')

            with torch.no_grad():
                model.eval()
                val_losses = []
                for X_val, y_val in test_loader:
                    X_val, y_val = X_val.to(device), y_val.to(device).unsqueeze(1)

                    y_val_pred = model(X_val)

                    val_loss = rmsle(y_val, y_val_pred)
                    val_losses.append(val_loss.item())

                avg_val_loss = np.mean(val_losses)
                print(f"Value Loss: {avg_val_loss}")

                if avg_val_loss < val_loss_min:
                    print(f'Val Loss Decreased: {val_loss_min} -> {val_loss}')
                    stop_cnt = 0
                    torch.save(model.state_dict(), "state_dict.pth")
                    val_loss_min = avg_val_loss
                else:
                    print(f"Val Loss remained the same")
                    stop_cnt += 1
                if stop_cnt >= stop_criteria:
                    print("Early Stopping")
                    break

            print(f"Best Val Loss: {val_loss_min}")



    ############################################################
    #    Test Data Preprocessing
    ############################################################

    df = pd.read_csv("./test.csv", index_col='id')

    sex_dummies = pd.get_dummies(df['Sex'], prefix='Sex')
    df = pd.concat([df, sex_dummies], axis=1)
    df.drop(['Sex'], axis=1, inplace=True)

    X_test = torch.tensor(df.values.astype(np.float32)).unsqueeze(1)

    test_dataset = TensorDataset(X_test)
    test_data_loader = DataLoader(test_dataset, shuffle=False)

    ############################################################
    #    Produce Test Output from Model
    ############################################################

    test_model = Regressor().to(device)
    test_model.load_state_dict(torch.load('./state_dict.pth'))

    test_model.eval()


    predictions = list()
    for X_batch, in test_data_loader:
        X_batch = X_batch[0].to(device)
        out = test_model(X_batch).detach().cpu().numpy()
        predictions.append(out)

    predictions = np.concatenate(predictions).flatten()
    predictions = np.round(predictions).astype(int)

    print(predictions)
    out_df = pd.DataFrame({
        'id': pd.read_csv('./test.csv')['id'],
        'Rings': predictions  # Assuming the output needs to be flattened
    })
    out_df.to_csv('./result.csv', index=False)

    print(out_df.head(40))

if __name__ == "__main__":
    main(train_model=True)
