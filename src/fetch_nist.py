import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

def fetch_nist_property(name):
    try:
        url = f"https://webbook.nist.gov/cgi/cbook.cgi?Name={name}&Units=SI"
        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'html.parser')

        # Example: Extract critical temp & pressure
        Tc, Pc, omega = None, None, None
        table = soup.find("table")
        if table:
            for row in table.find_all("tr"):
                cells = [c.get_text(strip=True) for c in row.find_all("td")]
                if "Critical temperature" in row.text:
                    Tc = cells[1].split()[0]
                elif "Critical pressure" in row.text:
                    Pc = cells[1].split()[0]
                elif "Acentric factor" in row.text:
                    omega = cells[1]

        return {"Name": name, "Tc_K": Tc, "Pc_MPa": Pc, "AcentricFactor": omega}
    except Exception as e:
        print(f"NIST fetch error for {name}: {e}")
        return {"Name": name, "Tc_K": None, "Pc_MPa": None, "AcentricFactor": None}

if __name__ == "__main__":
    with open("data/chemicals.txt") as f:
        chemicals = [line.strip() for line in f if line.strip()]

    results = [fetch_nist_property(name) for name in chemicals]
    df_nist = pd.DataFrame(results)
    df_nist.to_csv("data/nist_data.csv", index=False)
    print("✅ NIST data saved to data/nist_data.csv")

