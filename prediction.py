import Bio.SeqIO
import numpy as np
import pandas as pd
import torch
import Bio
import os
from create_data import *
from utils import *
from models.ginconv import GINConvNet

# Load the trained model
with open('model_GINConvNet_davis.model', 'rb') as file:
    model = torch.load(file)

# Prepare the input data
# write a csv file with the data
dataset = pd.DataFrame()
features = ['compound_iso_smiles', 'target_sequence', 'affinity']

# dictionaries mapping SMILES to PubChem ID and drug name
drugs = {
    'CN(C)C1=CC2=C(C=C1)N=C3C=CC(=[N+](C)C)C=C3S2.[Cl-]': ('methylene blue', '6099'),
    'COC1=C(C=CC(=C1)C=CC(=O)CC(=O)C=CC2=CC(=C(C=C2)O)OC)O': ('Curcumin', '969516'),
    'CN1CCC23C=CC(CC2OC4=C(C=CC(=C34)C1)OC)O.Br': ('Galantamine Hydrobromide', '121587'),
    'CN(C)CC1=NC2=C(C=C1)C(=CC(=C2O)Cl)Cl': ('5,7-dichloro-2-[(dimethylamino)methyl]quinolin-8-ol', '10016012'),
    'CCCN(CCN1CCN(CC1)C2=CC=C(C=C2)C3=CC(=C(C=C3)O)O)C4CCC5=C(C4)SC(=N5)N': ('Tau-aggregation-IN-1', '163408861'),
    'C1CN(CCN1C2=CC(=C(C=C2)[N+](=O)[O-])NC3=CC=CC=C3)C(=O)C4=CC=CC=C4': ('Abeta/tau aggregation-IN-3', '44814403'),
}

drugs2 = {
    'CC(=O)OC1=CC=CC=C1C(=O)O': ('Aspirin', '2244'),
    'C(C1C(C(C(C(O1)O)O)O)O)O': ('D-Glucose', '5793'),
}

def load_fasta(file):
    try:
        with open(file, 'r') as file:
            sequence = file.read()
        return sequence
    except Exception as e:
        print(f"Error loading fasta file: {e}")

target = '5O3L' # tau protein paired helical filament in Alzheimer's disease brain
target_sequence = load_fasta(f'data/prediction_data/rcsb_pdb_{target}.fasta') # load the fasta file
target_sequence = Bio.SeqIO.read(f'data/prediction_data/rcsb_pdb_{target}.fasta', 'fasta').seq
print(f"Target sequence: {str(target_sequence)}")

def create_data(drugs, target_sequence, dataset='input'):
    # Create dataframe with input data
    filename = f'data/{dataset}_data.csv'
    df = pd.DataFrame(columns=['compound_iso_smiles', 'target_sequence'])

    for smiles in drugs.keys():
        # Append data to the dataframe
        df = df._append({
            'compound_iso_smiles': smiles,
            'target_sequence': target_sequence,
            # 'affinity': 0.0  # Placeholder for affinity
        }, ignore_index=True)

    # save data to csv
    df.to_csv(filename, index=False)
    return df


def load_csv(filename):
    try:
        with open(filename, 'r') as filename:
            data = pd.read_csv(filename)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")


def preprocess_data(df=None, dataset='input', multi_target=False):
    """
    Preprocess the input data
    Input data is a csv file with columns: 'compound_iso_smiles', 'target_sequence'
    """
    print(f"Processing data: '{dataset}'")
    drugs, prots = list(df['compound_iso_smiles']), list(df['target_sequence'])

    smile_graph = {}
    for smile in df['compound_iso_smiles']:
        g = smile_to_graph(smile)
        smile_graph[smile] = g

    if multi_target:
        encoded_prots = [seq_cat(p) for p in prots]
    else:
        encoded_prots = [seq_cat(prots[0])] * len(drugs)
    drugs, prots = np.asarray(drugs), np.asarray(encoded_prots) # Convert to numpy arrays
    return drugs, prots


class EvaluationDataset(TestbedDataset):
    def __init__(self, root='/tmp', dataset='input', drugs=None, prots=None,
                 xd=None, xt=None, y=None, transform=None,
                 pre_transform=None,smile_graph=None):

        #root is required for save preprocessed data, default is '/tmp'
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.drugs = drugs
        self.prots = prots
        self.process(xd, xt, smile_graph)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def process(self, xd, xt, smile_graph):
        assert len(xd) == len(xt), "The number of drugs and targets must be the same!"
        data_list = []
        data_len = len(xd)
        for i in range(data_len):
            smiles = xd[i]
            target = xt[i]
            c_size, features, edge_index = smile_graph[smiles]
            GCNData = DATA.Data(x=torch.Tensor(np.array(features)),
                        edge_index=torch.LongTensor(edge_index).transpose(1, 0))
            GCNData.target = torch.LongTensor(np.array(target))
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            data_list.append(GCNData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

def make_prediction(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = None  # Unlabeled dataset
    print('Making predictions for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
    return total_labels, total_preds.numpy().flatten()

# make a dataframe with the predictions
def save_as_csv(df, predictions, data: EvaluationDataset):
    exp, preds = predictions
    assert len(df) == len(preds), f"The number of predictions do not match the number of samples: {len(preds)} predictions vs. {len(df)} samples"

    # copy the data to a new dataframe
    new_df = df.copy()

    # add predictions to the dataframe as a new column: 'predicted_affinity'
    if exp:
        new_df['expected_affinity'] = exp

    new_df['predicted_affinity'] = preds
    print(f"\nPredictions for dataset '{data.dataset}':")
    print(new_df.head(10))

    best_result = new_df.loc[new_df['predicted_affinity'].idxmax()]
    name, id = data.drugs.get(best_result['compound_iso_smiles'])
    print(f"\nMaximum predicted affinity found for '{name}' (PubChem id {id}): {np.max(preds)}")
    print(f"Average predicted affinity: {np.mean(preds)}")

    filename = f'predictions/{data.dataset}_prediction.csv'
    print(f"\nPredictions saved to file: {filename}")

    new_df.to_csv(filename, index=False)

    return new_df


if __name__ == '__main__':
    # Create the input data
    datasets = {
        'input': drugs,
        'input2': drugs2,
    }

    for dataset, drugs in datasets.items():
        print(f"------------------------------------------------------------")
        df = create_data(drugs, target_sequence, dataset)
        drugs, prots = list(df['compound_iso_smiles']), list(df['target_sequence'])

        smile_graph = {}
        for smile in drugs:
            g = smile_to_graph(smile)
            smile_graph[smile] = g

        drugs, prots = preprocess_data(df)
        # print("Preprocessed data:", drugs, prots)
        processed_data = EvaluationDataset(root='data', dataset=dataset, drugs=datasets[dataset], prots=prots, xd=drugs, xt=prots, smile_graph=smile_graph)

        # Prepare model
        model_state_dict = torch.load('model_GINConvNet_davis.model', map_location='cpu')
        model = GINConvNet()  # Replace YourModelClass with the class of your model
        model.load_state_dict(model_state_dict)

        # Make predictions
        loader = DataLoader(processed_data, batch_size=1, shuffle=False)
        pred = make_prediction(model, 'cpu', loader)
        save_as_csv(df, pred, processed_data)
