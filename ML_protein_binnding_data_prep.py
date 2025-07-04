# Install missing modules
!pip install rdkit --quiet
!pip install pubchempy --quiet



 # Data Science libraries
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import requests

# Get ML libraries
from transformers import AutoTokenizer, AutoModel
import torch

# data manipulation
import pickle

# parallel library
from concurrent.futures import ProcessPoolExecutor, as_completed

# add time to measure speed
import time

# Get RDkit libraries
from rdkit import Chem
from rdkit.Chem import AllChem
import pubchempy as pcp
from rdkit.Chem import AllChem, rdMolDescriptors, Descriptors



# save requirements
!pip freeze > requirements_data_prep.txt



# read the data (test)
data=pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Deloitte_project/Data/Deloitte_DrugDiscovery_dataset.csv')
# show the data
data.head(5)



# keep only reliably estimated values
data=data[data['kiba_score_estimated']==True]



# basic properties
n_points=data.shape[0]
n_features=data.shape[1]
print(f'Number of points: {n_points}')
print(f'Number of features: {n_features}')



n_unique_uniport=data['UniProt_ID'].nunique()
print(f'Number of unique UniPort IDs: {n_unique_uniport}')
n_unique_pubchem=len(np.unique(data['pubchem_cid']))
print(f'Number of unique PubChem IDs: {n_unique_pubchem}')



print('Original dataset')
print(data.shape)
print('No NaN dataset')
print(data.dropna().shape)

# remove the NaNs from the shuffled data
data=data.dropna()



# Find duplicates based on UniProt_ID and pubchem_cid
duplicates = data[data.duplicated(subset=['UniProt_ID', 'pubchem_cid'], keep=False)]

# Display the duplicate rows
print("Duplicate pairs based on UniProt_ID and pubchem_cid:")
print(duplicates.shape)
duplicates.head(5)



# get values for unique samples
UniProt_ID_unique=np.unique(duplicates['UniProt_ID'].values)

# Plot kiba_score for multiple unique UniPort
n_sample=[0, 1, 2, 3 ,4, 5, 6, 7]

fig, axes = plt.subplots(2, 4, figsize=(12, 8))

# Flatten axes for easy indexing in the loop
axes = axes.flatten()

# Plot values for 4 scores
for i, sample in enumerate(n_sample):
    # Filter the data for the specific UniProt_ID
    data_to_plot = duplicates[duplicates['UniProt_ID'] == UniProt_ID_unique[sample]]['kiba_score'].values
    # Plot the histogram in the corresponding subplot
    axes[i].hist(data_to_plot, bins=30, edgecolor='black', alpha=0.7)
    axes[i].set_xlabel('KIBA Score')
    axes[i].set_ylabel('Frequency')
    axes[i].set_title(f'KIBA Scores for {UniProt_ID_unique[sample]}')
# no overlap
plt.tight_layout()



# Calculate the mean kiba_score for each pair and assign it back to each row in the group
data['kiba_score_mean'] = data.groupby(['UniProt_ID', 'pubchem_cid'])['kiba_score'].transform('mean')

# Drop the original kiba_score column and remove duplicates based on unique pairs
data_unique = data.drop(columns='kiba_score').drop_duplicates(subset=['UniProt_ID', 'pubchem_cid']).rename(columns={'kiba_score_mean': 'kiba_score'})

# Display the result
print(data_unique)



# check for duplicates in the data
has_duplicates = data_unique.duplicated(subset=['UniProt_ID', 'pubchem_cid']).any()
print('Are there duplicates?')
print(has_duplicates)



# find median for kiba_score
median_thr = np.median(data_unique['kiba_score'].values)
print('Mean value of kiba_score')
print(median_thr)



# Calculate the median threshold for the KIBA score
thr = 0.01

# Create the histogram with log scale on the horizontal axis
plt.figure(figsize=(10, 6))
sns.histplot(data_unique['kiba_score'], bins=30, log_scale=(True, False))
plt.axvline(thr, color='red', linestyle='dashed', linewidth=2, label=f'Threshold: {thr:.2f}')
plt.xlabel('KIBA Score (log scale)')
plt.title('Histogram of KIBA Scores with Threshold')
plt.legend()
plt.show()



# MANUAL THRESHOLD

# Add label for bounded based on median
# 0 - not found, 1 - bound
data_unique['bound'] = (data_unique['kiba_score'] > 0).astype(int)



# create negative data by shuffling
data_negative = data_unique.copy()

# shuffle only pubchem_id, preserve the order for uniport
data_negative['pubchem_cid'] = np.random.permutation(data_negative['pubchem_cid'].values)

# set kiba score for zero (synthetic negative samples)

# added kiba_score as higher than medium -> not bound
data_negative['kiba_score']= 0
data_negative['kiba_score_estimated']=False

# make sure these pairs are negative
data_negative['bound']=0

# show results
print('Negative samples')
print(data_negative.head(5))
print()
print('Positive samples')
print(data_unique.head(5))




# check if there are no randomly occuring pairs due to random shuffling
positive_pairs = set(tuple(x) for x in data_unique[['UniProt_ID', 'pubchem_cid']].to_numpy())
negative_pairs = set(tuple(x) for x in data_negative[['UniProt_ID', 'pubchem_cid']].to_numpy())

# find common pairs
common_pairs = positive_pairs.intersection(negative_pairs)

# check if the intersection is empty
if len(common_pairs) == 0:
       print("No randomly occurring pairs found.")
else:
  print(f"Found {len(common_pairs)} randomly occurring pairs:")
  print(common_pairs)




# Convert common_pairs to a list of lists for easier filtering
common_pairs_list = [list(pair) for pair in common_pairs]

# Filter data_negative using a boolean mask
data_negative_filtered = data_negative[~data_negative[['UniProt_ID', 'pubchem_cid']].apply(tuple, axis=1).isin(common_pairs)]



# check again that there are no negative pairs in filtered dataset
positive_pairs = set(tuple(x) for x in data_unique[['UniProt_ID', 'pubchem_cid']].to_numpy())

# Create a set of negative pairs from the filtered DataFrame
negative_pairs_filtered = set(tuple(x) for x in data_negative_filtered[['UniProt_ID', 'pubchem_cid']].to_numpy())

# Find the intersection
common_pairs_filtered = positive_pairs.intersection(negative_pairs_filtered)

# Check if the intersection is empty
if len(common_pairs_filtered) == 0:
    print("No randomly occurring pairs found in data_negative_filtered.")
else:
    print(f"Found {len(common_pairs_filtered)} randomly occurring pairs in data_negative_filtered:")
    print(common_pairs_filtered)



# Concatenate the two DataFrames
combined_data = pd.concat([data_unique, data_negative_filtered], ignore_index=True)
# Display the result
print(combined_data)



# shuffle the order of rows in the dataframe
shuffled_data = combined_data.sample(frac=1, random_state=37).reset_index(drop=True)
# show the dataset
shuffled_data



# Create the histogram using seaborn
sns.histplot(x=shuffled_data['bound'])  # Use histplot for histogram

# Set plot labels and title
plt.xlabel('Bound (1) / Not Bound (0)')  # Update x-axis label
plt.ylabel('Frequency')  # Add y-axis label
plt.title('Histogram of Bound vs. Not Bound')  # Update title



# save intermediate results
shuffled_data.to_csv('/content/drive/MyDrive/Colab Notebooks/Deloitte_project/Data/combined_data.csv', index=False)


# reload data
shuffled_data=pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Deloitte_project/Data/combined_data.csv')



# get the data for ProtBert embeddings
bert_data_test=np.load('/content/drive/MyDrive/Colab Notebooks/Deloitte_project/Data/Pro_bert_test_embeddings.npy')
bert_data_train=np.load('/content/drive/MyDrive/Colab Notebooks/Deloitte_project/Data/Pro_bert_train_embeddings.npy')

# dictionary of IDs
bert_data_test_ids=np.load('/content/drive/MyDrive/Colab Notebooks/Deloitte_project/Data/Prot_Bert_test_ids.npy')
bert_data_train_ids=np.load('/content/drive/MyDrive/Colab Notebooks/Deloitte_project/Data/Prot_Bert_train_ids.npy')

# concatenate arrays
bert_data_ids=np.concatenate((bert_data_test_ids, bert_data_train_ids), axis=0)
bert_data=np.concatenate((bert_data_test, bert_data_train), axis=0)

# cobmine existing protein embeddings
protein_embeddings=dict(zip(bert_data_ids, bert_data))



common_proteins = set(protein_embeddings.keys()).intersection(set(shuffled_data['UniProt_ID']))
num_common_proteins = len(common_proteins)
print('UnProt_id vs CIFAR Kaggle')
print()
print(f"Number of common proteins: {num_common_proteins}")
print()

different_proteins = set(np.unique(shuffled_data['UniProt_ID'].values)) - set(protein_embeddings.keys())
num_different_proteins = len(different_proteins)
print(f"Number of different proteins: {num_different_proteins}")




def fetch_protein_sequence(uniprot_id):
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
    response = requests.get(url)
    if response.status_code == 200:
        fasta_data = response.text
        sequence = ''.join(fasta_data.split('\n')[1:])
        return sequence
    else:
        print(f"Error fetching sequence for {uniprot_id}")
        return None


def get_protein_embedding(sequence):

    # Tokenize the sequence
    inputs = tokenizer(sequence, return_tensors="pt", is_split_into_words=False, padding=True, truncation=True, add_special_tokens=True)

    # Move inputs to CUDA
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state

    return embeddings


def get_protein_embedding_fixed(sequence, max_length=1024):
    """
    Generates a protein embedding from a given sequence using Prot-Bert.

    Args:
        sequence (str): The protein sequence.
        max_length (int, optional): The maximum sequence length for padding/truncation. Defaults to 1024.

    Returns:
        torch.Tensor: The protein embedding with a fixed size.
    """

    # Tokenize the sequence
    inputs = tokenizer(sequence, return_tensors="pt", is_split_into_words=False, padding='max_length', truncation=True, add_special_tokens=True, max_length=max_length)

    # Move inputs to CUDA
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state

    # Mean pooling to get a fixed-size embedding
    embeddings = torch.mean(embeddings, dim=1)  # Average across the sequence dimension

    return embeddings




# Initialize ProtBERT model and tokenizer and move model to CUDA
model_name = "Rostlab/prot_bert_bfd"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Move the model to CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Example UniProt IDs and embedding storage
uniprot_ids = shuffled_data['UniProt_ID'].values

# Get unique values only for uniprot_ids
uniprot_ids = np.unique(uniprot_ids)

# create a list of missed protein embeddings
protein_embeddings_missed = {}


# Generate embeddings: for all files

# convert different proteins to list
different_proteins_list = list(different_proteins)

start_time = time.time()

n_prots=50

for uniprot_id in different_proteins_list[:n_prots]:
    # Get amino acid sequence
    sequence = fetch_protein_sequence(uniprot_id)

    # Add spaces to the sequence for proper tokenization
    sequence = " ".join(sequence)

    if sequence:
        # Get embeddings
        embeddings = get_protein_embedding_fixed(sequence)

        # Move embeddings back to CPU and convert to numpy
        protein_embeddings_missed[uniprot_id] = embeddings.view(-1).cpu().numpy()
    else:
        print(f"No sequence found for UniProt_ID: {uniprot_id}")

end_time = time.time()

inference_time = end_time - start_time

print(f"Inference time: {inference_time} seconds" + ' ' + str(n_prots) + ' proteins')



# save missed protein embeddings
# save to pickle
with open('/content/drive/MyDrive/Colab Notebooks/Deloitte_project/Data/protein_embeddings_missed.pkl', 'wb') as f:
    pickle.dump(protein_embeddings_missed, f)



# save to pickle
with open('/content/drive/MyDrive/Colab Notebooks/Deloitte_project/Data/protein_embeddings.pkl', 'wb') as f:
    pickle.dump(protein_embeddings, f)



# reload from pickle
with open('/content/drive/MyDrive/Colab Notebooks/Deloitte_project/Data/protein_embeddings.pkl', 'rb') as f:
    protein_embeddings = pickle.load(f)



# read cid_smiles map
cid_smiles_df=pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Deloitte_project/Data/CID-SMILES.txt', sep=None, engine='python', header=None)
cid_smiles_df.head(5)



# add column headers
cid_smiles_df.columns = ['pubchem_cid', 'smiles']



# get pubchem ids from shuffled data
pubchem_ids = np.unique(shuffled_data['pubchem_cid'].values)
pubchem_cids = list(pubchem_ids.astype(int))

# create a dictionary for cid to smiles
cid_to_smiles = dict(zip(cid_smiles_df['pubchem_cid'], cid_smiles_df['smiles']))



# save dictionary
with open('/content/drive/MyDrive/Colab Notebooks/Deloitte_project/Data/cid_to_smiles.pkl', 'wb') as f:
    pickle.dump(cid_to_smiles, f)



# reload the dictionary
with open('/content/drive/MyDrive/Colab Notebooks/Deloitte_project/Data/cid_to_smiles.pkl', 'rb') as f:
    cid_to_smiles = pickle.load(f)



def process_ligand_batch(pub_ids):

    '''Function to process ligands in batches based on molecular cid'''

    batch_results = {}

    for pub_id in pub_ids:
        smiles = cid_to_smiles.get(pub_id, None)
        if smiles is None:
            continue

        # Convert SMILES to RDKit molecule object
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue  # Skip invalid molecules

        # Calculate molecular fingerprint and descriptors
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        fingerprint_bits = list(fingerprint)

        # save all other parameters
        molecular_weight = Descriptors.MolWt(mol)
        logP = Descriptors.MolLogP(mol)
        polar_surface_area = Descriptors.TPSA(mol)
        num_rotatable_bonds = Descriptors.NumRotatableBonds(mol)
        num_h_donors = Descriptors.NumHDonors(mol)
        num_h_acceptors = Descriptors.NumHAcceptors(mol)
        num_aromatic_rings = Descriptors.NumAromaticRings(mol)
        fraction_csp3 = Descriptors.FractionCSP3(mol)
        bertz_complexity = Descriptors.BertzCT(mol)

        # Store results in dictionary for this batch
        batch_results[pub_id] = {
            'Fingerprint': fingerprint_bits,
            'SMILES': smiles,
            'MolecularWeight': molecular_weight,
            'LogP': logP,
            'PolarSurfaceArea': polar_surface_area,
            'NumRotatableBonds': num_rotatable_bonds,
            'NumHDonors': num_h_donors,
            'NumHAcceptors': num_h_acceptors,
            'NumAromaticRings': num_aromatic_rings,
            'FractionCSP3': fraction_csp3,
            'BertzComplexity': bertz_complexity
        }

    return batch_results



# Dictionary for batch of ligands
ligand_data_600 = {}

# Prepare pubchem IDs and SMILES mapping for faster access
pubchem_ids = np.unique(shuffled_data['pubchem_cid'].values)
pubchem_cids = list(pubchem_ids.astype(int))

# set up the last batch
pubchem_cids=pubchem_cids[500000:]

# Parallel execution with batch processing
start_time = time.time()
batch_size = 50  # Adjust batch size based on your system’s capabilities

# Split pubchem_cids into batches
batches = [pubchem_cids[i:i + batch_size] for i in range(0, len(pubchem_cids), batch_size)]

# Use ProcessPoolExecutor for parallel processing of batches
with ProcessPoolExecutor() as executor:
    futures = {executor.submit(process_ligand_batch, batch): batch for batch in batches}
    for future in as_completed(futures):
        result_batch = future.result()
        ligand_data_600.update(result_batch)  # Aggregate results into main dictionary

end_time = time.time()
print(f"Total time taken: {end_time - start_time} seconds")




print('Number of ligands processed')
print(len(ligand_data_600))


# 1st batch

# save ligand data to pickle file
with open('/content/drive/MyDrive/Colab Notebooks/Deloitte_project/Data/ligand_data_100.pkl', 'wb') as f:
    pickle.dump(ligand_data_100, f)


# 2nd batch

# save ligand data to pickle file
with open('/content/drive/MyDrive/Colab Notebooks/Deloitte_project/Data/ligand_data_200.pkl', 'wb') as f:
    pickle.dump(ligand_data_200, f)


# 3nd batch

# save ligand data to pickle file
with open('/content/drive/MyDrive/Colab Notebooks/Deloitte_project/Data/ligand_data_300.pkl', 'wb') as f:
    pickle.dump(ligand_data_300, f)


# 4th batch

# save ligand data to pickle file
with open('/content/drive/MyDrive/Colab Notebooks/Deloitte_project/Data/ligand_data_400.pkl', 'wb') as f:
    pickle.dump(ligand_data_400, f)


# 5th batch

# save ligand data to pickle file
with open('/content/drive/MyDrive/Colab Notebooks/Deloitte_project/Data/ligand_data_500.pkl', 'wb') as f:
    pickle.dump(ligand_data_500, f)


# 6th batch

# save ligand data to pickle file
with open('/content/drive/MyDrive/Colab Notebooks/Deloitte_project/Data/ligand_data_600.pkl', 'wb') as f:
    pickle.dump(ligand_data_600, f)



# Load ligand data from the pickle file
with open('/content/drive/MyDrive/Colab Notebooks/Deloitte_project/Data/ligand_data_100.pkl', 'rb') as f:
    ligand_data_100 = pickle.load(f)


# Load ligand data from the pickle file
with open('/content/drive/MyDrive/Colab Notebooks/Deloitte_project/Data/ligand_data_200.pkl', 'rb') as f:
    ligand_data_200 = pickle.load(f)



# Load ligand data from the pickle file
with open('/content/drive/MyDrive/Colab Notebooks/Deloitte_project/Data/ligand_data_300.pkl', 'rb') as f:
    ligand_data_300 = pickle.load(f)



# Load ligand data from the pickle file
with open('/content/drive/MyDrive/Colab Notebooks/Deloitte_project/Data/ligand_data_400.pkl', 'rb') as f:
    ligand_data_400 = pickle.load(f)



# Load ligand data from the pickle file
with open('/content/drive/MyDrive/Colab Notebooks/Deloitte_project/Data/ligand_data_500.pkl', 'rb') as f:
    ligand_data_500 = pickle.load(f)



# Load ligand data from the pickle file
with open('/content/drive/MyDrive/Colab Notebooks/Deloitte_project/Data/ligand_data_600.pkl', 'rb') as f:
    ligand_data_600 = pickle.load(f)



# combined dict for all data
ligang_data_all={}

# save all dict data
for d in [ligand_data_100, ligand_data_200, ligand_data_300, ligand_data_400, ligand_data_500, ligand_data_600]:
    ligang_data_all.update(d)



# Save combined data
with open('/content/drive/MyDrive/Colab Notebooks/Deloitte_project/Data/ligang_data.pkl', 'wb') as f:
    pickle.dump(ligang_data_all, f)



