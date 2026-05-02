from fastapi import APIRouter, Query
from typing import Optional

router = APIRouter()

SCHEMES = [
    {
        "id": "pm_kisan", "name": "PM-KISAN", "ministry": "Ministry of Agriculture",
        "benefit": "₹6,000/year direct income support",
        "benefit_amount": 6000, "type": "Subsidy",
        "eligibility_criteria": {"land_size_max": 2, "farmer_type": "small"},
        "description": "Direct income support of ₹6000 per year to small farmers in 3 installments",
        "documents": ["Aadhaar Card", "Bank Account", "Land Records"],
        "apply_link": "https://pmkisan.gov.in",
        "crops": ["all"], "states": ["all"],
        "simple_explanation": "Government gives ₹6000 every year directly to your bank account if you have small farm land.",
    },
    {
        "id": "fasal_bima", "name": "PM Fasal Bima Yojana", "ministry": "Ministry of Agriculture",
        "benefit": "Crop insurance up to full sum insured",
        "benefit_amount": 50000, "type": "Insurance",
        "eligibility_criteria": {"land_size_max": 100, "farmer_type": "all"},
        "description": "Crop insurance scheme providing financial support if crops fail due to natural calamities",
        "documents": ["Aadhaar", "Bank Account", "Land Records", "Sowing Certificate"],
        "apply_link": "https://pmfby.gov.in",
        "crops": ["rice","wheat","maize","cotton","sugarcane"],
        "states": ["all"],
        "simple_explanation": "If your crop is destroyed by flood, drought or disease, government pays compensation.",
    },
    {
        "id": "kisan_credit", "name": "Kisan Credit Card", "ministry": "RBI / NABARD",
        "benefit": "Low interest loan up to ₹3 lakh at 4% interest",
        "benefit_amount": 300000, "type": "Loan",
        "eligibility_criteria": {"land_size_max": 100, "farmer_type": "all"},
        "description": "Short-term credit for agricultural needs at subsidized interest rates",
        "documents": ["Aadhaar", "Pan Card", "Land Records", "Bank Account"],
        "apply_link": "https://www.nabard.org",
        "crops": ["all"], "states": ["all"],
        "simple_explanation": "Get a special bank card to borrow money for seeds, fertilizers at very low 4% interest.",
    },
    {
        "id": "soil_health_card", "name": "Soil Health Card Scheme", "ministry": "Ministry of Agriculture",
        "benefit": "Free soil testing and fertilizer recommendation",
        "benefit_amount": 0, "type": "Training",
        "eligibility_criteria": {"land_size_max": 100, "farmer_type": "all"},
        "description": "Free soil testing to know nutrient status and get customized fertilizer recommendations",
        "documents": ["Aadhaar", "Land Records"],
        "apply_link": "https://soilhealth.dac.gov.in",
        "crops": ["all"], "states": ["all"],
        "simple_explanation": "Government tests your soil for free and gives a card showing which fertilizers to use.",
    },
    {
        "id": "drip_irrigation", "name": "Pradhan Mantri Krishi Sinchayee Yojana",
        "ministry": "Ministry of Jal Shakti",
        "benefit": "55% subsidy on drip/sprinkler irrigation",
        "benefit_amount": 75000, "type": "Subsidy",
        "eligibility_criteria": {"land_size_max": 5, "farmer_type": "small"},
        "description": "Subsidy on micro-irrigation systems to improve water use efficiency",
        "documents": ["Aadhaar", "Land Records", "Bank Account", "Quotation from dealer"],
        "apply_link": "https://pmksy.gov.in",
        "crops": ["all"], "states": ["all"],
        "simple_explanation": "Get 55% discount when buying drip or sprinkler irrigation system for your farm.",
    },
    {
        "id": "agri_infra_fund", "name": "Agriculture Infrastructure Fund",
        "ministry": "Ministry of Agriculture",
        "benefit": "Low interest loan for farm infrastructure",
        "benefit_amount": 2000000, "type": "Loan",
        "eligibility_criteria": {"land_size_max": 100, "farmer_type": "all"},
        "description": "Medium to long-term debt financing for post-harvest management infrastructure",
        "documents": ["Aadhaar", "Pan Card", "Project Report", "Land Records"],
        "apply_link": "https://agriinfra.dac.gov.in",
        "crops": ["all"], "states": ["all"],
        "simple_explanation": "Get big loans to build warehouses, cold storage, or processing units on your farm.",
    },
]

@router.get("/get_schemes")
async def get_schemes(
    crop: Optional[str] = Query(None),
    state: Optional[str] = Query(None),
    land_size: Optional[float] = Query(None),
    scheme_type: Optional[str] = Query(None),
):
    results = []
    for scheme in SCHEMES:
        # Filter by crop
        if crop and scheme["crops"] != ["all"]:
            if crop.lower() not in [c.lower() for c in scheme["crops"]]:
                continue
        # Filter by type
        if scheme_type and scheme["type"].lower() != scheme_type.lower():
            continue
        # Check eligibility
        eligible = True
        reason = "You are eligible!"
        if land_size:
            max_size = scheme["eligibility_criteria"].get("land_size_max", 100)
            if land_size > max_size:
                eligible = False
                reason = f"Land size must be ≤ {max_size} acres"
        results.append({**scheme, "eligible": eligible, "eligibility_reason": reason})

    return {
        "schemes": results,
        "total": len(results),
        "filters_applied": {"crop": crop, "state": state, "land_size": land_size}
    }

@router.get("/check_eligibility/{scheme_id}")
async def check_eligibility(
    scheme_id: str,
    land_size: float = Query(...),
    farmer_type: str = Query("small"),
):
    scheme = next((s for s in SCHEMES if s["id"] == scheme_id), None)
    if not scheme:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Scheme not found")

    eligible = True
    reasons = []
    max_size = scheme["eligibility_criteria"].get("land_size_max", 100)
    if land_size > max_size:
        eligible = False
        reasons.append(f"❌ Your land ({land_size} acres) exceeds limit of {max_size} acres")
    else:
        reasons.append(f"✅ Land size qualifies ({land_size} acres ≤ {max_size} acres)")

    if scheme["eligibility_criteria"].get("farmer_type") not in ["all", farmer_type]:
        eligible = False
        reasons.append(f"❌ This scheme is for {scheme['eligibility_criteria']['farmer_type']} farmers only")
    else:
        reasons.append("✅ Farmer type qualifies")

    return {
        "scheme_id": scheme_id,
        "scheme_name": scheme["name"],
        "eligible": eligible,
        "reasons": reasons,
        "required_documents": scheme["documents"],
        "apply_link": scheme["apply_link"],
        "benefit": scheme["benefit"],
        "simple_explanation": scheme["simple_explanation"],
    }
