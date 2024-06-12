import pandas as pd
import numpy as np
import os
import json,pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from utils import *
from torch.utils.data import random_split

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    [atom.GetIsAromatic()])

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)

    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append( feature / sum(feature) )

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    return c_size, features, edge_index

def seq_cat(prot):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]):
        x[i] = seq_dict[ch]
    return x

def train_test_split(df, test_split=0.2):
    """
    Split the dataset into training and testing sets
    """
    test_size = int(test_split * len(df))
    train_size = len(df) - test_size
    train_set, test_set = random_split(df, [train_size, test_size])

    train_df = df.iloc[train_set.indices]
    test_df = df.iloc[test_set.indices]

    return train_df, test_df


def process_deepDTA_data():
    all_prots = []
    input_datasets = ['kiba','davis']
    for input_data in input_datasets:
        if os.path.exists('data/processed/' + input_data + '_train.pt') and os.path.exists('data/processed/' + input_data + '_test.pt'):
            print(f"Processed data for '{input_data}' already exists")
            continue
        if not os.path.exists(f'data/{input_data}_train.csv') or not os.path.exists(f'data/{input_data}_test.csv'):
            print('convert data from DeepDTA for ', input_data)
            fpath = 'data/' + input_data + '/'
            train_fold = json.load(open(fpath + "folds/train_fold_setting1.txt"))
            train_fold = [ee for e in train_fold for ee in e ]
            valid_fold = json.load(open(fpath + "folds/test_fold_setting1.txt"))
            ligands = json.load(open(fpath + "ligands_can.txt"), object_pairs_hook=OrderedDict)
            proteins = json.load(open(fpath + "proteins.txt"), object_pairs_hook=OrderedDict)
            affinity = pickle.load(open(fpath + "Y","rb"), encoding='latin1')
            ligands = []
            prots = []
            for d in ligands.keys():
                lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]),isomericSmiles=True)
                ligands.append(lg)
            for t in proteins.keys():
                prots.append(proteins[t])
            if input_data == 'davis':
                affinity = [-np.log10(y/1e9) for y in affinity]
            affinity = np.asarray(affinity)
            opts = ['train','test']
            for opt in opts:
                rows, cols = np.where(np.isnan(affinity)==False)
                if opt=='train':
                    rows,cols = rows[train_fold], cols[train_fold]
                elif opt=='test':
                    rows,cols = rows[valid_fold], cols[valid_fold]
                with open('data/' + input_data + '_' + opt + '.csv', 'w') as f:
                    f.write('compound_iso_smiles,target_sequence,affinity\n')
                    for pair_ind in range(len(rows)):
                        ls = []
                        ls += [ ligands[rows[pair_ind]]  ]
                        ls += [ prots[cols[pair_ind]]  ]
                        ls += [ affinity[rows[pair_ind],cols[pair_ind]]  ]
                        f.write(','.join(map(str,ls)) + '\n')
            print('\ndataset:', input_data)
            print('train_fold:', len(train_fold))
            print('test_fold:', len(valid_fold))
            print('len(set(drugs)),len(set(prots)):', len(set(ligands)),len(set(prots)))
            all_prots += list(set(prots))
        else:
            print(f"data for '{input_data}' already exists")


        compound_iso_smiles = []
        for dt_name in ['kiba','davis']:
            opts = ['train','test']
            for opt in opts:
                # check if the processed data already exists
                if os.path.exists('data/processed/' + dt_name + '_' + opt + '.csv'):
                    continue
                df = pd.read_csv('data/' + dt_name + '_' + opt + '.csv')
                compound_iso_smiles += list( df['compound_iso_smiles'] )
        compound_iso_smiles = set(compound_iso_smiles)
        smile_graph = {}
        for smile in compound_iso_smiles:
            g = smile_to_graph(smile)
            smile_graph[smile] = g

        input_datasets = ['davis','kiba']
        # convert to PyTorch data format
        for input_data in input_datasets:
            processed_data_file_train = 'data/processed/' + input_data + '_train.pt'
            processed_data_file_test = 'data/processed/' + input_data + '_test.pt'
            if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
                df = pd.read_csv('data/' + input_data + '_train.csv')
                train_drugs, train_prots,  train_Y = list(df['compound_iso_smiles']),list(df['target_sequence']),list(df['affinity'])
                XT = [seq_cat(t) for t in train_prots]
                train_drugs, train_prots,  train_Y = np.asarray(train_drugs), np.asarray(XT), np.asarray(train_Y)
                df = pd.read_csv('data/' + input_data + '_test.csv')
                test_drugs, test_prots,  test_Y = list(df['compound_iso_smiles']),list(df['target_sequence']),list(df['affinity'])
                XT = [seq_cat(t) for t in test_prots]
                test_drugs, test_prots,  test_Y = np.asarray(test_drugs), np.asarray(XT), np.asarray(test_Y)

                # make data PyTorch Geometric ready
                print('preparing ', input_data + '_train.pt in pytorch format!')
                train_data = TestbedDataset(root='data', dataset=input_data+'_train', xd=train_drugs, xt=train_prots, y=train_Y,smile_graph=smile_graph)
                print('preparing ', input_data + '_test.pt in pytorch format!')
                test_data = TestbedDataset(root='data', dataset=input_data+'_test', xd=test_drugs, xt=test_prots, y=test_Y,smile_graph=smile_graph)
                print(processed_data_file_train, ' and ', processed_data_file_test, ' have been created')
            else:
                print(processed_data_file_train, ' and ', processed_data_file_test, ' are already created')


seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v:(i+1) for i,v in enumerate(seq_voc)}
seq_dict_len = len(seq_dict)
max_seq_len = 1000

amyloid_sequences = {
    'Ab40': 'DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVV', # amyloid beta 1-40
    'aS': 'MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGG', # alpha-synuclein
    'Ab42': 'DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA', # amyloid beta 1-42
    'Ab': 'MLPGLALLLLAAWTARALEVPTDGNAGLLAEPQIAMFCGRLNMHMNVQNGKWDSDPSGTKTCIDTKEGILQYCQEVYPELQITNVVEANQPVTIQNWCKRGRKQCKTHPHFVIPYRCLVGEFVSDALLVPDKCKFLHQERMDVCETHLHWHTVAKETCSEKSTNLHDYGMLLPCGIDKFRGVEFVCCPLAEESDNVDSADAEEDDSDVWWGGADTDYADGSEDKVVEVAEEEEVAEVEEEEADDDEDDEDGDEVEEEAEEPYEEATERTTSIATTTTTTTESVEEVVREVCSEQAETGPCRAMISRWYFDVTEGKCAPFFYGGCGGNRNNFDTEEYCMAVCGSAMSQSLLKTTQEPLARDPVKLPTTAASTPDAVDKYLETPGDENEHAHFQKAKERLEAKHRERMSQVMREWEEAERQAKNLPKADKKAVIQHFQEKVESLEQEAANERQQLVETHMARVEAMLNDRRRLALENYITALQAVPPRPRHVFNMLKKYVRAEQKDRQHTLKHFEHVRMVDPKKAAQIRSQVMTHLRVIYERMNQSLSLLYNVPAVAEEIQDEVDELLQKEQNYSDDVLANMISEPRISYGNDALMPSLTETKTTVELLPVNGEFSLDDLQPWHSFGADSVPANTENEVEPVDARPAADRGLTTRPGSGLTNIKTEEISEVKMDAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIATVIVITLVMLKKKQYTSIHHGVVEVDAAVTPEERHLSKMQQNGYENPTYKFFEQMQN', # amyloid beta
    't': 'VQIVYKPVDLSKVTSKCGSLGNIHHKPGGGQVEVKSEKLDFKDRVQSKIGSLDNITHVPGGGNKKIETHKLTF', # tau
}

def process_amyloid_data():
    if os.path.exists('data/processed/amyloid_train.pt') and os.path.exists('data/processed/amyloid_test.pt'):
        print(f"Processed data for 'amyloid' already exists")
        return

    # Load data
    df = pd.read_csv('data/amyloids/amyloid_data.csv')
    # Reduce the dataset to only the columns SMILES, Target fibril, and log(Kd/M)
    smiles = list(df['SMILES'])
    sequences = [amyloid_sequences.get(p, amyloid_sequences['Ab40']) for p in df['Target fibril']]
    new_df = pd.DataFrame({'smiles': smiles, 'sequence': sequences, 'log(Kd/M)': list(df['log(Kd/M)'])})
    new_df = new_df.dropna()

    if not os.path.exists(f'data/amyloid_train.csv') or not os.path.exists(f'data/amyloid_test.csv'):
        # Split data into train and test sets
        train_df, test_df = train_test_split(new_df, test_split=0.2)

        train_df.to_csv('data/amyloid_train.csv', index=False)
        test_df.to_csv('data/amyloid_test.csv', index=False)

    smile_graph = {
        smile: smile_to_graph(smile) for smile in smiles
    }
    for opt in ['train', 'test']:
        df = pd.read_csv('data/amyloid_' + opt + '.csv')

        drugs = np.asarray(df['smiles'])
        prots = np.asarray([seq_cat(s) for s in df['sequence']])
        Y = np.asarray([-float(n) for n in df['log(Kd/M)']])

        # make data PyTorch Geometric ready
        print(f'\npreparing amyloid_{opt}.pt in pytorch format!')
        data = TestbedDataset(root='data', dataset=f'amyloid_{opt}', xd=drugs, xt=prots, y=Y, smile_graph=smile_graph)
        print(f'processed/amyloid_{opt}.pt has been created')



if __name__ == '__main__':
    process_deepDTA_data()
    process_amyloid_data()
