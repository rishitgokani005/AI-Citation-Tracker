import json
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Union
import config
import db_manager

VALID_STAGES = ["research", "comparison", "recommendation", "problem_solving", "vendor_selection"]


def _atomic_write_json(filepath: Path, data: Union[List[Dict[str, Any]], List[str]]) -> None:
    """Writes data to a temporary file and atomically replaces target filepath."""
    filepath = Path(filepath)
    parent_dir = filepath.parent
    parent_dir.mkdir(parents=True, exist_ok=True)

    # Create a temporary file in the same directory for atomic replace
    temp_file = tempfile.NamedTemporaryFile("w", dir=parent_dir, delete=False, encoding="utf-8")
    temp_path = Path(temp_file.name)

    try:
        json.dump(data, temp_file, indent=2, ensure_ascii=False)
        temp_file.flush()
        os.fsync(temp_file.fileno())
        temp_file.close()

        # Validate that the temp file can be parsed back
        with open(temp_path, "r", encoding="utf-8") as check_f:
            json.load(check_f)

        # Atomic replace
        os.replace(temp_path, filepath)
    except Exception as err:
        if temp_path.exists():
            temp_path.unlink()
        raise IOError(f"Failed to atomically write JSON to {filepath}: {err}") from err


def load_prompts(prompts_path: Path = config.PROMPTS_PATH) -> List[Dict[str, Any]]:
    """Loads and validates prompts.json. Returns empty list if missing or empty."""
    prompts_path = Path(prompts_path)

    if not prompts_path.exists():
        return []

    try:
        with open(prompts_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return []
            data = json.loads(content)
            if not isinstance(data, list):
                raise ValueError(f"{prompts_path} does not contain a JSON array.")
            return data
    except json.JSONDecodeError as e:
        raise ValueError(f"Corrupted or invalid JSON in {prompts_path}: {e}") from e


def add_prompt(text: str, stage: str, prompts_path: Path = config.PROMPTS_PATH) -> Dict[str, Any]:
    """Appends a new prompt to prompts.json with auto-incremented ID and duplicate check."""
    clean_text = text.strip() if text else ""
    if not clean_text:
        raise ValueError("Prompt text cannot be empty.")

    clean_stage = stage.strip().lower() if stage else ""
    if clean_stage not in VALID_STAGES:
        raise ValueError(f"Invalid stage '{stage}'. Valid stages are: {', '.join(VALID_STAGES)}")

    existing_prompts = load_prompts(prompts_path)

    # Check for duplicate prompt text (case-insensitive, trimmed)
    clean_lower = clean_text.lower()
    for item in existing_prompts:
        if isinstance(item, dict) and item.get("text", "").strip().lower() == clean_lower:
            raise ValueError(f"Prompt '{clean_text}' already exists (ID #{item.get('id')}).")

    # Generate next ID
    existing_ids = [p["id"] for p in existing_prompts if isinstance(p, dict) and isinstance(p.get("id"), int)]
    next_id = max(existing_ids, default=0) + 1

    new_prompt = {
        "id": next_id,
        "text": clean_text,
        "stage": clean_stage
    }

    existing_prompts.append(new_prompt)
    _atomic_write_json(prompts_path, existing_prompts)
    return new_prompt


def update_prompt(prompt_id: int, text: str, stage: str, prompts_path: Path = config.PROMPTS_PATH) -> Dict[str, Any]:
    """Updates an existing prompt's text and stage in prompts.json."""
    clean_text = text.strip() if text else ""
    if not clean_text:
        raise ValueError("Prompt text cannot be empty.")

    clean_stage = stage.strip().lower() if stage else ""
    if clean_stage not in VALID_STAGES:
        raise ValueError(f"Invalid stage '{stage}'. Valid stages are: {', '.join(VALID_STAGES)}")

    existing_prompts = load_prompts(prompts_path)
    clean_lower = clean_text.lower()

    # Check for duplicate prompt text under a different ID
    for item in existing_prompts:
        if isinstance(item, dict) and item.get("id") != prompt_id:
            if item.get("text", "").strip().lower() == clean_lower:
                raise ValueError(f"Another prompt with text '{clean_text}' already exists (ID #{item.get('id')}).")

    found = False
    updated_prompt = {}
    for item in existing_prompts:
        if isinstance(item, dict) and item.get("id") == prompt_id:
            item["text"] = clean_text
            item["stage"] = clean_stage
            updated_prompt = item
            found = True
            break

    if not found:
        raise ValueError(f"Prompt ID #{prompt_id} not found.")

    _atomic_write_json(prompts_path, existing_prompts)
    return updated_prompt


def delete_prompt(prompt_id: int, prompts_path: Path = config.PROMPTS_PATH) -> bool:
    """Removes a prompt by ID from prompts.json."""
    existing_prompts = load_prompts(prompts_path)
    filtered = [p for p in existing_prompts if isinstance(p, dict) and p.get("id") != prompt_id]

    if len(filtered) == len(existing_prompts):
        raise ValueError(f"Prompt ID #{prompt_id} not found.")

    _atomic_write_json(prompts_path, filtered)
    return True


def delete_all_prompts(prompts_path: Path = config.PROMPTS_PATH) -> bool:
    """Deletes all prompts from prompts.json."""
    _atomic_write_json(prompts_path, [])
    return True



def load_brands(brands_path: Path = config.BRANDS_PATH) -> List[str]:
    """Loads and validates brands.json. Returns empty list if missing or empty."""
    brands_path = Path(brands_path)

    if not brands_path.exists():
        return []

    try:
        with open(brands_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return []
            data = json.loads(content)
            if not isinstance(data, list):
                raise ValueError(f"{brands_path} does not contain a JSON array.")
            return data
    except json.JSONDecodeError as e:
        raise ValueError(f"Corrupted or invalid JSON in {brands_path}: {e}") from e


def add_brand(name: str, brands_path: Path = config.BRANDS_PATH) -> str:
    """Appends a new brand to brands.json with duplicate check."""
    clean_name = name.strip() if name else ""
    if not clean_name:
        raise ValueError("Brand name cannot be empty.")

    existing_brands = load_brands(brands_path)

    # Check for duplicate brand name (case-insensitive, trimmed)
    clean_lower = clean_name.lower()
    for item in existing_brands:
        if isinstance(item, str) and item.strip().lower() == clean_lower:
            raise ValueError(f"Brand '{clean_name}' already exists.")

    existing_brands.append(clean_name)
    _atomic_write_json(brands_path, existing_brands)
    return clean_name


def update_brand(old_name: str, new_name: str, brands_path: Path = config.BRANDS_PATH) -> str:
    """Renames an existing brand in brands.json."""
    clean_new = new_name.strip() if new_name else ""
    if not clean_new:
        raise ValueError("Brand name cannot be empty.")

    clean_old = old_name.strip() if old_name else ""
    existing_brands = load_brands(brands_path)

    old_lower = clean_old.lower()
    new_lower = clean_new.lower()

    # Verify old name exists
    if not any(isinstance(b, str) and b.strip().lower() == old_lower for b in existing_brands):
        raise ValueError(f"Brand '{old_name}' not found.")

    # Verify new name does not conflict with another existing brand
    for item in existing_brands:
        if isinstance(item, str) and item.strip().lower() != old_lower:
            if item.strip().lower() == new_lower:
                raise ValueError(f"Brand '{clean_new}' already exists.")

    updated_brands = []
    for item in existing_brands:
        if isinstance(item, str) and item.strip().lower() == old_lower:
            updated_brands.append(clean_new)
        else:
            updated_brands.append(item)

    _atomic_write_json(brands_path, updated_brands)
    return clean_new


def delete_brand(name: str, brands_path: Path = config.BRANDS_PATH) -> bool:
    """Deletes a brand from brands.json."""
    clean_lower = name.strip().lower() if name else ""
    existing_brands = load_brands(brands_path)

    filtered = [b for b in existing_brands if isinstance(b, str) and b.strip().lower() != clean_lower]

    if len(filtered) == len(existing_brands):
        raise ValueError(f"Brand '{name}' not found.")

    _atomic_write_json(brands_path, filtered)
    return True


def delete_all_brands(brands_path: Path = config.BRANDS_PATH) -> bool:
    """Deletes all brands from brands.json."""
    _atomic_write_json(brands_path, [])
    return True



def prompt_has_responses(prompt_id: int, db_path=None) -> bool:
    """Checks if historical DB records exist for a prompt_id."""
    target_db = db_path or config.DB_PATH
    if not Path(target_db).exists():
        return False

    try:
        conn = db_manager.get_connection(target_db)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM responses WHERE prompt_id = ?", (prompt_id,))
            row = cursor.fetchone()
            return row[0] > 0 if row else False
        finally:
            conn.close()
    except Exception:
        return False


def brand_has_mentions(brand_name: str, db_path=None) -> bool:
    """Checks if historical DB mention records exist for brand_name."""
    target_db = db_path or config.DB_PATH
    if not Path(target_db).exists():
        return False

    try:
        conn = db_manager.get_connection(target_db)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM mentions WHERE LOWER(brand_name) = LOWER(?)", (brand_name.strip(),))
            row = cursor.fetchone()
            return row[0] > 0 if row else False
        finally:
            conn.close()
    except Exception:
        return False
