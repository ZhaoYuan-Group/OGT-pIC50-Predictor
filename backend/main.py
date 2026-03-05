# -*- coding: utf-8 -*-
import io
import uuid
import datetime
import traceback
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, Descriptors, DataStructs
from DeepPurpose import utils, CompoundPred

# ========================== Path Configuration ==========================
CURRENT_FILE_PATH = Path(__file__).resolve()
BACKEND_DIR = CURRENT_FILE_PATH.parent
MODEL_ROOT = BACKEND_DIR.parent / "model"  

if not MODEL_ROOT.exists():
    raise FileNotFoundError(f"The model root directory does not exist, please check the path: {MODEL_ROOT}")
print(f"The model root directory has been confirmed: {MODEL_ROOT}")

# ========================== Model Encoding Mapping ==========================
MODEL_DEFAULT_ENCODING = {
    "rdkit_2d_normalizedModel": "rdkit_2d_normalized",
    "DaylightModel": "Daylight",
    "ErGModel": "ErG",
    "MorganModel": "Morgan"
}

# ========================== Pydantic Model Definitions ==========================
class ResultItem(BaseModel):
    id: str
    smiles: str
    pic50: float
    model_used: str
    timestamp: str
    molecule_name: str
    mol_wt: float
    logp: float
    hbd: int
    hba: int


class PaginatedResults(BaseModel):
    total_items: int
    total_pages: int
    current_page: int
    items: List[ResultItem]
    model_used: Optional[str]  # Return the currently filtered model name

# ========================== Multi-Model Management ==========================
class ModelManager:
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.configs: Dict[str, Any] = {}
        self.default_model: str = None

    def load_model(self, model_name: str, config_filename: str, model_filename: str) -> None:
        try:
            print(f"\nLoading: {model_name}...")
            model_dir = MODEL_ROOT / model_name
            if not model_dir.exists():
                raise FileNotFoundError(f"Model folder does not exist: {model_dir}")
            
            config_path = model_dir / config_filename
            model_path = model_dir / model_filename
            
            if not config_path.exists():
                raise FileNotFoundError(f"The configuration file does not exist: {config_path}")
            if not model_path.exists():
                raise FileNotFoundError(f"The model file does not exist: {model_path}")

            # Load configuration and model
            config = utils.load_dict(str(model_dir))
            model = CompoundPred.model_initialize(**config)
            model.load_pretrained(str(model_path))

            # Ensure drug_encoding exists in configuration
            if "drug_encoding" not in config:
                config["drug_encoding"] = MODEL_DEFAULT_ENCODING.get(model_name, "rdkit_2d_normalized")
                print(f"Model {model_name} configuration supplemented with drug_encoding: {config['drug_encoding']}")

            self.models[model_name] = model
            self.configs[model_name] = config
            print(f"Model {model_name} loaded successfully")

            if self.default_model is None:
                self.default_model = model_name

        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_name}: {e}")

    def get_model(self, model_name: Optional[str] = None) -> Any:
        if not self.models:
            raise RuntimeError("No models loaded")
        
        target_model_name = model_name or self.default_model
        if target_model_name not in self.models:
            raise ValueError(f"Model {target_model_name} does not exist, available models: {list(self.models.keys())}")
        return self.models[target_model_name]

    def get_model_config(self, model_name: Optional[str] = None) -> Any:
        if not self.configs:
            raise RuntimeError("No model configurations loaded")
            
        target_model_name = model_name or self.default_model
        if target_model_name not in self.configs:
            raise ValueError(f"Configuration for model {target_model_name} does not exist")
        return self.configs[target_model_name]

    def list_models(self) -> List[str]:
        return list(self.models.keys())

# ========================== Model Loading ==========================
try:
    print("Initializing model manager...")
    model_manager = ModelManager()
    
    # Load all models
    model_manager.load_model("rdkit_2d_normalizedModel", "config.pkl", "model.pt")
    model_manager.load_model("DaylightModel", "config.pkl", "model.pt")
    model_manager.load_model("ErGModel", "config.pkl", "model.pt")
    model_manager.load_model("MorganModel", "config.pkl", "model.pt")
    
    print(f"\nAll models loaded, available models: {model_manager.list_models()}")
    print(f"Default model: {model_manager.default_model}")

except Exception as e:
    raise RuntimeError(f"Failed to load models on startup: {e}")

# ========================== In-Memory Database Optimization ==========================
# Store results by model for improved query efficiency
db: Dict[str, dict] = {}  # Global storage for all results
model_results: Dict[str, List[str]] = {}  # Record result IDs for each model

# Initialize result lists for each model
for model_name in model_manager.list_models():
    model_results[model_name] = []

# ========================== Molecular Feature Generation Functions ==========================
def smiles_to_rdkit2d(smiles_list: List[str]) -> np.ndarray:
    """Convert SMILES to RDKit 2D features"""
    features = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            desc = [
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.TPSA(mol)
            ]
            features.append(desc)
        else:
            features.append([0.0]*5)
    return np.array(features, dtype=np.float32)

def smiles_to_morgan(smiles_list: List[str], radius=2, nBits=2048) -> np.ndarray:
    """Convert SMILES to Morgan fingerprints"""
    features = []
    # Handle RDKit version differences (GetMorganGenerator/MorganGenerator)
    try:
        generator = AllChem.MorganGenerator(radius=radius, nBits=nBits)
    except AttributeError:
        generator = AllChem.GetMorganGenerator(radius=radius, nBits=nBits)
        
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fp = generator.GetFingerprint(mol)
            arr = np.zeros((nBits,), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr)
            features.append(arr)
        else:
            features.append(np.zeros(nBits, dtype=np.float32))
    return np.array(features)

def smiles_to_daylight(smiles_list: List[str]) -> np.ndarray:
    """Convert SMILES to Daylight fingerprints"""
    features = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fp = AllChem.RDKFingerprint(
                mol,
                maxPath=5,
                fpSize=2048,
                useHs=True,
                tgtDensity=0.0,
                minPath=1
            )
            arr = np.zeros((2048,), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr)
            features.append(arr)
        else:
            features.append(np.zeros(2048, dtype=np.float32))
    return np.array(features)

def smiles_to_erg(smiles_list: List[str]) -> np.ndarray:
    """Convert SMILES to ErG features"""
    features = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            features.append([
                mol.GetNumAtoms(),
                mol.GetNumBonds(),
                Descriptors.MolWt(mol)
            ])
        else:
            features.append([0.0, 0.0, 0.0])
    return np.array(features, dtype=np.float32)

# ========================== FastAPI Application Initialization ==========================
app = FastAPI(title="pIC50 Predictor API", version="2.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"],
)

# ========================== Helper Functions ==========================
def run_prediction(smiles_list: List[str], model_name: Optional[str] = None) -> List[ResultItem]:
    """Execute prediction and return list of ResultItem"""
    if not smiles_list:
        return []
        
    # Filter valid SMILES
    valid_smiles = [s for s in smiles_list if Chem.MolFromSmiles(s) is not None]
    invalid_smiles = [s for s in smiles_list if s not in valid_smiles]
    if invalid_smiles:
        print(f"Warning: Detected {len(invalid_smiles)} invalid SMILES strings, skipped")
    if not valid_smiles:
        raise HTTPException(status_code=400, detail="No valid SMILES strings were provided.")
        
    try:
        # Get model and configuration
        model = model_manager.get_model(model_name)
        model_config = model_manager.get_model_config(model_name)
        used_model = model_name or model_manager.default_model
        print(f"Using model {used_model} for predictions")

        # Get encoding method
        drug_encoding = model_config.get(
            "drug_encoding", 
            MODEL_DEFAULT_ENCODING.get(used_model, "rdkit_2d_normalized")
        )
        print(f"Model encoding method: {drug_encoding}")

        # Build input data (using DeepPurpose standard processing pipeline)
        df_input = pd.DataFrame({
            "Drug": valid_smiles,
            "Label": [0.0] * len(valid_smiles)  # Placeholder labels
        })
        
        x_pred = utils.data_process(
            X_drug=df_input["Drug"].values,
            y=df_input["Label"].values,
            drug_encoding=drug_encoding,
            split_method="no_split"
        )

        # Execute prediction
        predictions = model.predict(x_pred)
        
        # Unify prediction result format to numpy array
        if isinstance(predictions, list):
            predictions = np.array(predictions, dtype=np.float32).flatten()
        elif not isinstance(predictions, np.ndarray):
            raise TypeError(f"Abnormal prediction result type: {type(predictions)}")

        print(f"Prediction completed, result shape: {predictions.shape}, Data Type: {predictions.dtype}")

        # Verify number of prediction results
        if len(predictions) != len(valid_smiles):
            raise ValueError(
                f"Number of prediction results does not match (input: {len(valid_smiles)}, output: {len(predictions)})"
            )

        # Build ResultItem list
        current_time = datetime.datetime.now().isoformat()
        results = []
        for i, (smiles, pred_value) in enumerate(zip(valid_smiles, predictions)):
            try:
                mol = Chem.MolFromSmiles(smiles)
                pic50 = round(float(pred_value), 3)
                results.append(ResultItem(
                    id=str(uuid.uuid4()),
                    smiles=smiles,
                    pic50=pic50,
                    model_used=used_model,
                    timestamp=current_time,
                    molecule_name=f"Molecule-{len(db) + i + 1}",
                    mol_wt=Descriptors.MolWt(mol),
                    logp=Descriptors.MolLogP(mol),
                    hbd=Descriptors.NumHDonors(mol),
                    hba=Descriptors.NumHAcceptors(mol)
                ))
            except (TypeError, ValueError) as e:
                raise HTTPException(
                    status_code=500, 
                    detail=f"Invalid predicted value '{pred_value}' (SMILES: {smiles}): {str(e)}"
                )

        return results

    except Exception as e:
        error_detail = f"Prediction failure: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)
        raise HTTPException(status_code=500, detail=error_detail)

# ========================== API Endpoint Definitions ==========================
@app.get("/")
def read_root():
    return {
        "message": "Welcome to the pIC50 Prediction API",
        "available_models": model_manager.list_models(),
        "default_model": model_manager.default_model
    }

@app.post("/predict", response_model=List[ResultItem], tags=["predict"])
def predict_single(
    smiles: str = Query(..., description="SMILES string (supports multiple lines, one per line)"),
    model_name: Optional[str] = Query(None, description="Model name, available values: " + ", ".join(model_manager.list_models()))
):
    """Predict pIC50 values for single or multiple SMILES strings"""
    smiles_list = [s.strip() for s in smiles.split("\n") if s.strip()]
    if not smiles_list:
        raise HTTPException(status_code=422, detail="Please provide at least one valid SMILES string.")
    
    results = run_prediction(smiles_list, model_name)
    
    # Save results to in-memory database and record result IDs for the model
    for item in results:
        db[item.id] = item.dict()
        # Add result ID to corresponding model's list
        if item.model_used not in model_results:
            model_results[item.model_used] = []
        model_results[item.model_used].append(item.id)
        
    return results

@app.post("/upload_csv", response_model=List[ResultItem], tags=["predict"])
async def upload_csv(
    file: UploadFile = File(..., description="CSV file containing a SMILES column"),
    model_name: Optional[str] = Form(None, description="Model name, available values: " + ", ".join(model_manager.list_models()))
):
    """Upload SMILES from CSV file and predict pIC50 values"""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type, please upload a CSV file.")
    
    contents = await file.read()
    try:
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        if 'SMILES' not in df.columns:
            raise HTTPException(status_code=400, detail="The CSV file must contain a 'SMILES' column.")
        smiles_list = df['SMILES'].dropna().tolist()
        print(f"Extracted up to {len(smiles_list)} SMILES strings from CSV")
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV file: {str(e)}")

    results = run_prediction(smiles_list, model_name)
    
    # Save results to in-memory database and record result IDs for the model
    for item in results:
        db[item.id] = item.dict()
        # Add result ID to corresponding model's list
        if item.model_used not in model_results:
            model_results[item.model_used] = []
        model_results[item.model_used].append(item.id)
        
    return results

@app.get("/results", response_model=PaginatedResults, tags=["result"])
def get_results(
    page: int = Query(1, ge=1, description="Page number, starting from 1"), 
    size: int = Query(10, ge=1, description="Number of items displayed per page"),
    model_name: Optional[str] = Query(None, description="Filter results by model name. Use 'all' to get all results."),
    sort_by: Optional[str] = Query("timestamp", description="Field to sort by (e.g., 'pic50', 'mol_wt', 'timestamp')"),
    sort_dir: Optional[str] = Query("desc", description="Sort direction: 'asc' for ascending, 'desc' for descending")
):
    """Query historical prediction results with pagination, support filtering by model and dynamic sorting"""
    # Filter results by model
    if model_name and model_name.lower() != 'all':
        # Validate model name exists
        if model_name not in model_manager.list_models():
            raise HTTPException(status_code=400, detail=f"Model {model_name} does not exist")
        
        # Get all result IDs for this model and corresponding results
        result_ids = model_results.get(model_name, [])
        all_items = [db[result_id] for result_id in result_ids if result_id in db]
    else:
        # Get results from all models
        all_items = list(db.values())
    
    # Dynamic sorting logic
    if all_items:
        # Check if sort field exists to prevent errors
        sample_item = all_items[0]
        if sort_by not in sample_item:
            # Fallback to default sorting by timestamp if field is invalid
            sort_by = "timestamp"
            
        reverse_order = sort_dir.lower() == "desc"
        
        # Sort full dataset
        all_items.sort(key=lambda x: x.get(sort_by, 0), reverse=reverse_order)
    
    # Calculate pagination information
    total_items = len(all_items)
    total_pages = (total_items + size - 1) // size or 1
    start_index = (page - 1) * size
    paginated_items = all_items[start_index:start_index + size]
    
    return PaginatedResults(
        total_items=total_items,
        total_pages=total_pages,
        current_page=page,
        items=[ResultItem(**item) for item in paginated_items],
        model_used=model_name if model_name and model_name.lower() != 'all' else None
    )

@app.get("/plot_distribution", tags=["Visualization"])
def get_plot_distribution(model_name: Optional[str] = Query(None, description="Filter by model name")):
    """Get histogram data for pIC50 prediction value distribution"""
    if model_name:
        # Use model result index to get all prediction values for this model
        result_ids = model_results.get(model_name, [])
        predictions = [db[result_id]['pic50'] for result_id in result_ids if result_id in db]
    else:
        predictions = [item['pic50'] for item in db.values()]

    if not predictions:
        return {"labels": [], "values": [], "model_used": model_name or "all"}
        
    hist, bin_edges = np.histogram(predictions, bins=10, range=(0, 10))
    labels = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(len(bin_edges)-1)]
    
    return {
        "labels": labels,
        "values": hist.tolist(),
        "model_used": model_name or "all"
    }

@app.get("/mol_image", tags=["Visualization"])
def get_mol_image(smiles: str = Query(..., description="SMILES string used to generate the molecular structure image")):
    """Generate PNG image of molecular structure"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise HTTPException(status_code=400, detail="Invalid SMILES string, unable to generate molecular structure.")
    
    img = Draw.MolToImage(mol, size=(250, 250))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return Response(content=buffer.getvalue(), media_type="image/png")

@app.get("/models", tags=["Model Management"])
def get_available_models():
    """Get list of all available models"""
    return {
        "available_models": model_manager.list_models(),
        "default_model": model_manager.default_model
    }