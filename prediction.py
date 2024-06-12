import Bio.SeqIO
import numpy as np
import pandas as pd
import torch
import Bio
import os
from create_data import *
from utils import *

from models.ginconv import GINConvNet
from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet

import requests
import pubchempy as pcp
import torch_scatter as ts


def load_fasta(file):
    try:
        with open(file, 'r') as file:
            sequence = file.read()
        return sequence
    except Exception as e:
        print(f"Error loading fasta file: {e}")


def create_data(drugs, target_sequence, dataset='input'):
    """
    Create dataframe with input data
    """
    filename = f'data/{dataset}_data.csv'
    df = pd.DataFrame(columns=['compound_iso_smiles', 'target_sequence'])

    for smiles in drugs.keys():
        # Append data to the dataframe
        df = df._append({
            'compound_iso_smiles': smiles,
            'target_sequence': target_sequence,
        }, ignore_index=True)

    # save data to csv
    df.to_csv(filename, index=False)
    return df


def pubchem_id_lookup(pubchem_id):
    """
    Get the name and SMILES of a molecule given its PubChem ID
    """
    try:
        compound = pcp.Compound.from_cid(pubchem_id)
        try: # try to get the first synonym
            return compound.synonyms[0]
        except IndexError: # if no synonyms found, return the IUPAC name
            return compound.iupac_name
    except Exception as e:
        print(f"Error looking up PubChem ID {pubchem_id}: {e}")
        return None


def get_name_from_chembl_id(chembl_id):
    url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/{chembl_id}.json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        try: # check for preferred name
            return data['pref_name']
        except KeyError:
            try: # check for synonyms
                return data['molecule_synonyms'][0]['molecule_synonym']
            except KeyError: # return CHEMBL id if no preferred name or synonyms found
                return chembl_id
    else:
        return None


def get_ligands_from(reference_dataset='davis'):
    """
    Returns a dictionary mapping drug SMILES to PubChem ID and drug name from the given dataset
    Saves the data to a csv file
    """
    if os.path.exists(f'data/{reference_dataset}/ligands_info.csv'):
        df = pd.read_csv(f'data/{reference_dataset}/ligands_info.csv')
        out = {}
        for i, row in df.iterrows():
            out[row['Canonical SMILES']] = row['Name'], row['PubChem ID']
        return out

    ligands = json.load(open(f"data/{reference_dataset}/ligands_can.txt"), object_pairs_hook=OrderedDict)

    out = {}
    df = pd.DataFrame(columns=['PubChem ID', 'Name', 'Canonical SMILES'])
    for id, ligand in ligands.items():
        name = pubchem_id_lookup(id) if reference_dataset == 'davis' else id
        out[ligand] = name, id
        df = df._append({
            'PubChem ID': id,
            'Name': name,
            'Canonical SMILES': ligand
        }, ignore_index=True)

    df.to_csv(f'data/{reference_dataset}/ligands_info.csv', index=False)

    return out


def load_csv(filename):
    try:
        with open(filename, 'r') as filename:
            data = pd.read_csv(filename)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")


def preprocess_data(df=None, dataset='input', multi_target=True):
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
    """
    Class for the evaluation dataset, inheriting from the TestbedDataset class
    """
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


def save_as_csv(df, predictions, data: EvaluationDataset, model_name="GINConvNet_davis"):
    exp, preds = predictions
    assert len(df) == len(preds), f"The number of predictions do not match the number of samples: {len(preds)} predictions vs. {len(df)} samples"

    # copy the data to a new dataframe
    new_df = df.copy()

    # add predictions to the dataframe as a new column: 'predicted_affinity'
    if exp:
        new_df['expected_affinity'] = exp

    new_df['predicted_affinity'] = preds

    num_best_results = 10
    print(f"\nTop {min(len(new_df), num_best_results)} out of {len(new_df)} predictions for dataset '{data.dataset}' sorted by affinity:")
    print(new_df.nlargest(num_best_results, 'predicted_affinity'))

    best_result = new_df.loc[new_df['predicted_affinity'].idxmax()]
    compound_name, id = data.drugs.get(best_result['compound_iso_smiles'])
    print(f"\nBest binding result:\tligand '{compound_name}' ({'PubChem' if data.dataset == 'davis' else 'CHEMBL'} id: {id}) and target {target}")
    print(f"Maximum predicted affinity = {best_result['predicted_affinity']}")
    print(f"Average predicted affinity = {np.mean(preds)}")
    with open('predictions/summary.csv', 'a') as f:
        f.write(f"{data.dataset}, {model_name}, {str(compound_name)}, {id}, {best_result['predicted_affinity']}, {best_result['compound_iso_smiles']}, predictions/{data.dataset}_{model_name}_pred.csv\n")

    filename = f'predictions/{data.dataset}_{model_name}_pred.csv'
    print(f"\nPredictions saved to file: {filename}")

    new_df.to_csv(filename, index=False)

    return new_df


def create_empty_model_instance(model_name):
    match model_name:
        case 'GINConvNet':
            return GINConvNet()
        case 'GATNet':
            return GATNet()
        case 'GAT_GCN':
            return GAT_GCN()
        case 'GCNNet':
            return GCNNet()
        case _:
            raise ValueError(f"Model '{model_name}' not found")



if __name__ == '__main__':
    # Create the input data (ligands and target sequence)
    # dictionaries mapping SMILES to PubChem ID and drug name
    ligands = {
        'CN(C)C1=CC2=C(C=C1)N=C3C=CC(=[N+](C)C)C=C3S2.[Cl-]': ('methylene blue', '6099'),
        'COC1=C(C=CC(=C1)C=CC(=O)CC(=O)C=CC2=CC(=C(C=C2)O)OC)O': ('Curcumin', '969516'),
        'CN1CCC23C=CC(CC2OC4=C(C=CC(=C34)C1)OC)O.Br': ('Galantamine Hydrobromide', '121587'),
        'CN(C)CC1=NC2=C(C=C1)C(=CC(=C2O)Cl)Cl': ('5,7-dichloro-2-[(dimethylamino)methyl]quinolin-8-ol', '10016012'),
        'CCCN(CCN1CCN(CC1)C2=CC=C(C=C2)C3=CC(=C(C=C3)O)O)C4CCC5=C(C4)SC(=N5)N': ('Tau-aggregation-IN-1', '163408861'),
        'C1CN(CCN1C2=CC(=C(C=C2)[N+](=O)[O-])NC3=CC=CC=C3)C(=O)C4=CC=CC=C4': ('Abeta/tau aggregation-IN-3', '44814403'),
    }

    # control = {
    #     'CC(=O)OC1=CC=CC=C1C(=O)O': ('Aspirin', '2244'),
    #     'C(C1C(C(C(C(O1)O)O)O)O)O': ('D-Glucose', '5793'),
    # }

    all_davis = get_ligands_from('davis')
    all_kiba = get_ligands_from('kiba')

    input_datasets = {
        'input': ligands,
        # 'control': control,
        'davis': all_davis,
        'kiba': all_kiba,
    }

    # Load target sequence

    target = '5O3L' # tau protein paired helical filament in Alzheimer's disease brain
    target_sequence = Bio.SeqIO.read(f'data/test_targets/rcsb_pdb_{target}.fasta', 'fasta').seq
    print(f"\n{target} sequence: {target_sequence}")

    # target = '3IW4'
    # target_sequence = 'MPSEDRKQPSNNLDRVKLTDFNFLMVLGKGSFGKVMLADRKGTEELYAIKILKKDVVIQDDDVECTMVEKRVLALLDKPPFLTQLHSCFQTVDRLYFVMEYVNGGDLMYHIQQVGKFKEPQAVFYAAEISIGLFFLHKRGIIYRDLKLDNVMLDSEGHIKIADFGMCKEHMMDGVTTREFCGTPDYIAPEIIAYQPYGKSVDWWAYGVLLYEMLAGQPPFDGEDEDELFQSIMEHNVSYPKSLSKEAVSICKGLMTKHPAKRLGCGPEGERDVREHAFFRRIDWEKLENREIQPPFKPKVCGKGAENFDKFFTRGQPVLTPPDQLVIANIDQSDFEGFSYVNPQFVHPILQSAVHHHHHH'

    # make a new csv file with the summary
    with open('predictions/summary.csv', 'w') as f:
        f.write('Data, Model Name, Compound Name, PubChem id, Predicted Affinity, Canonical SMILES, filename\n')

    # Process and save each dataset, then predict binding affinities for each drug-target pair
    for input_data, ligands in input_datasets.items():
        print(f"------------------------------------------------------------")
        df = create_data(ligands, target_sequence, input_data)
        ligands, prots = list(df['compound_iso_smiles']), list(df['target_sequence'])

        smile_graph = {}
        for smile in ligands:
            g = smile_to_graph(smile)
            smile_graph[smile] = g

        # Preprocess the data
        ligands, prots = preprocess_data(df, input_data, multi_target=False)
        processed_data = EvaluationDataset(root='data', dataset=input_data, drugs=input_datasets[input_data], prots=prots, xd=ligands, xt=prots, smile_graph=smile_graph)

        # Load each of the 8 models and make predictions on the input dataset
        for model_name in ['GINConvNet', 'GATNet', 'GAT_GCN', 'GCNNet']:
            for training_dataset in ['davis', 'kiba']:
                print(f"\nPredicting binding affinities for dataset '{input_data}' using model '{model_name}' trained on {training_dataset}...")

                # Prepare models
                model_state_dict = torch.load(f'training_results/model_{model_name}_{training_dataset}.model', map_location='cpu')
                model = create_empty_model_instance(model_name)
                model.load_state_dict(model_state_dict)

                # Make predictions
                loader = DataLoader(processed_data, batch_size=1, shuffle=False)
                pred = make_prediction(model, 'cpu', loader)
                save_as_csv(df, pred, processed_data, model_name = model_name + '_' + training_dataset)
