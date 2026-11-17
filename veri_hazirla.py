import time
import pandas as pd
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem

def get_smiles_from_name(compound_name):
    """Molekul isminden PubChem araciligiyla Canonical SMILES kodunu ceker."""
    try:
        compounds = pcp.get_compounds(compound_name, 'name')
        if compounds and len(compounds) > 0:
            time.sleep(0.2)
            return compounds[0].canonical_smiles
    except Exception as e:
        print(f"Hata ({compound_name}): {e}")
    return None

def calculate_descriptors(smiles):
    """SMILES kodundan temel fizikokimyasal ozellikleri ve ECFP4 parmak izini hesaplar."""
    if not smiles:
        return None
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # 1. Temel Fizikokimyasal Tanimlayicilar
    descriptors = {
        'MolLogP': round(Descriptors.MolLogP(mol), 3),
        'MW': round(Descriptors.MolWt(mol), 2),
        'TPSA': round(Descriptors.TPSA(mol), 2),
        'HBD': Descriptors.NumHDonors(mol),
        'HBA': Descriptors.NumHAcceptors(mol),
        'RotBonds': Descriptors.NumRotatableBonds(mol),
        'HeavyAtoms': mol.GetNumHeavyAtoms()
    }
    
    # 2. Yapisal Temsil: 1024-bit Morgan Parmak Izi (ECFP4)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
    for i, bit in enumerate(fp):
        descriptors[f'ECFP4_{i}'] = int(bit)
        
    return descriptors

def process_dataset(csv_filename):
    print(f"\n--- {csv_filename} isleniyor ---")
    try:
        # on_bad_lines='skip' parametresi bozuk satirlarin hata vermesini engeller
        df = pd.read_csv(csv_filename, on_bad_lines='skip')
    except FileNotFoundError:
        print(f"Hata: {csv_filename} dosyasi bulunamadi!")
        return None

    print(f"1. PubChem'den SMILES kodlari cekiliyor...")
    df['smiles'] = df['compound_name'].apply(get_smiles_from_name)
    df = df.dropna(subset=['smiles']).reset_index(drop=True)

    print(f"2. RDKit ile fizikokimyasal ozellikler hesaplaniyor...")
    desc_list = df['smiles'].apply(calculate_descriptors)
    
    valid_mask = desc_list.notna()
    df = df[valid_mask].reset_index(drop=True)
    desc_df = pd.DataFrame(desc_list[valid_mask].tolist())

    final_df = pd.concat([df, desc_df], axis=1)
    return final_df

# --- ANA CALISTIRMA BLOGU ---
hayvan_df = process_dataset("hayvan_ham.csv")
bitki_df = process_dataset("bitki_ham.csv")

if hayvan_df is not None and bitki_df is not None:
    master_df = pd.concat([hayvan_df, bitki_df], ignore_index=True)
    master_df.to_csv("hazir_veri_seti.csv", index=False)
    
    print("\n========================================")
    print(f" ISLEM BASARIYLA TAMAMLANDI!")
    print(f" Toplam Hayvan Molekulu: {len(hayvan_df)}") 
    print(f" Toplam Bitki Molekulu: {len(bitki_df)}")
    print(f" Birlestirilmis Ana Set: {len(master_df)} molekul")
    print(" Kaydedilen dosya: hazir_veri_seti.csv")
    print("========================================")
    print(master_df[['compound_name', 'membrane_type', 'MolLogP', 'MW', 'permeability_value']].head(12))