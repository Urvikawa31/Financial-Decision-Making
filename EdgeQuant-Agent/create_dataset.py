from datasets import load_dataset
import pandas as pd
import json
import os


def load_clef_dataset(save_raw=True):
    print("Loading dataset from HuggingFace...")

    dataset = load_dataset("TheFinAI/CLEF_Task3_Trading")

    print("Available splits:", dataset)

    btc_df = dataset["BTC"].to_pandas()
    tsla_df = dataset["TSLA"].to_pandas()

    print("Dataset Loaded Successfully.")
    print("BTC Shape:", btc_df.shape)
    print("TSLA Shape:", tsla_df.shape)

    if save_raw:
        os.makedirs("data", exist_ok=True)

        # 🔥 FINAL SAFE CLEAN FUNCTION
        def clean_df(df):
            df = df.copy()
        
            def clean_value(x):
                if hasattr(x, "tolist"):
                    x = x.tolist()
        
                if x is None:
                    return None
        
                if isinstance(x, list):
                    return [str(i) for i in x]
        
                if isinstance(x, dict):
                    return {str(k): str(v) for k, v in x.items()}
        
                return str(x)
        
            for col in df.columns:
                df[col] = df[col].apply(clean_value)
        
            records = df.to_dict(orient="records")
        
            # 🔥 CONVERT LIST → DATE-KEY DICT (IMPORTANT)
            final_dict = {}
            for row in records:
                date = row["date"]
                final_dict[date] = row
        
            return final_dict

        btc_records = clean_df(btc_df)
        tsla_records = clean_df(tsla_df)

        with open("data/btc.json", "w", encoding="utf-8") as f:
            json.dump(btc_records, f, indent=2, ensure_ascii=False)

        with open("data/tsla.json", "w", encoding="utf-8") as f:
            json.dump(tsla_records, f, indent=2, ensure_ascii=False)

        print("✅ JSON files recreated successfully!")

    return btc_df, tsla_df


if __name__ == "__main__":
    load_clef_dataset(True)