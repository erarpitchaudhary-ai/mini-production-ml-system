
import pandas as pd


TRAIN_PATH = "data/processed/splits/train.csv.gz"
TEST_PATH = "data/processed/splits/test.csv.gz"


def calculate_event_distribution(df):
    return df["event"].value_counts(normalize=True)


def calculate_cold_start_rates(train, current):

    known_users = set(train["visitorid"].unique())
    known_items = set(train["itemid"].unique())

    new_user_rate = (
        ~current["visitorid"].isin(known_users)
    ).mean()

    new_item_rate = (
        ~current["itemid"].isin(known_items)
    ).mean()

    return new_user_rate * 100, new_item_rate * 100


def main():

    print("Loading chronological datasets...")

    train = pd.read_csv(TRAIN_PATH)
    current = pd.read_csv(TEST_PATH)

    print("\n========== EVENT DISTRIBUTION ==========")

    train_dist = calculate_event_distribution(train)
    current_dist = calculate_event_distribution(current)

    event_report = pd.DataFrame({
        "Training (%)": train_dist * 100,
        "Test (%)": current_dist * 100
    }).fillna(0)

    event_report["Absolute Change (pp)"] = (
        event_report["Test (%)"] - event_report["Training (%)"]
    ).abs()

    print(event_report.round(2))

    print("\n========== COLD-START MONITORING ==========")

    new_user_rate, new_item_rate = calculate_cold_start_rates(
        train, current
    )

    print(f"New user interaction rate : {new_user_rate:.2f}%")
    print(f"New product interaction rate : {new_item_rate:.2f}%")

    print("\n========== MONITORING SUMMARY ==========")

    print("Training interactions:", len(train))
    print("Test interactions:", len(current))

    print(
        "Unique training users:",
        train["visitorid"].nunique()
    )

    print(
        "Unique test users:",
        current["visitorid"].nunique()
    )

    print("\nMonitoring completed successfully.")

import pandas as pd

TRAIN_PATH = "data/processed/splits/train.csv.gz"
TEST_PATH = "data/processed/splits/test.csv.gz"


def calculate_event_distribution(df):
    return df["event"].value_counts(normalize=True)


def calculate_cold_start_rates(train, current):
    known_users = set(train["visitorid"].unique())
    known_items = set(train["itemid"].unique())

    new_user_rate = (
        ~current["visitorid"].isin(known_users)
    ).mean()

    new_item_rate = (
        ~current["itemid"].isin(known_items)
    ).mean()

    return new_user_rate * 100, new_item_rate * 100


# ADD THE NEW FUNCTION HERE
def generate_alerts(event_report, new_user_rate, new_item_rate):

    print("\n========== DRIFT ALERTS ==========")

    alerts = []

    if (event_report["Absolute Change (pp)"] > 1.0).any():
        alerts.append("WARNING: Event distribution drift detected.")

    if new_user_rate > 80:
        alerts.append("WARNING: High new-user interaction rate.")

    if new_item_rate > 15:
        alerts.append("WARNING: High new-product interaction rate.")

    if alerts:
        for alert in alerts:
            print(alert)
    else:
        print("No monitoring thresholds exceeded.")


# EXISTING MAIN FUNCTION
def main():

    print("Loading chronological datasets...")

    train = pd.read_csv(TRAIN_PATH)
    current = pd.read_csv(TEST_PATH)

    print("\n========== EVENT DISTRIBUTION ==========")

    train_dist = calculate_event_distribution(train)
    current_dist = calculate_event_distribution(current)

    event_report = pd.DataFrame({
        "Training (%)": train_dist * 100,
        "Test (%)": current_dist * 100
    }).fillna(0)

    event_report["Absolute Change (pp)"] = (
        event_report["Test (%)"] - event_report["Training (%)"]
    ).abs()

    print(event_report.round(2))

    print("\n========== COLD-START MONITORING ==========")

    new_user_rate, new_item_rate = calculate_cold_start_rates(
        train, current
    )

    print(f"New user interaction rate : {new_user_rate:.2f}%")
    print(f"New product interaction rate : {new_item_rate:.2f}%")

    # CALL THE NEW FUNCTION HERE
    generate_alerts(event_report, new_user_rate, new_item_rate)

    print("\n========== MONITORING SUMMARY ==========")

    print("Training interactions:", len(train))
    print("Test interactions:", len(current))

    print("Unique training users:", train["visitorid"].nunique())
    print("Unique test users:", current["visitorid"].nunique())

    print("\nMonitoring completed successfully.")


# KEEP THIS AT THE VERY END
if __name__ == "__main__":
    main()
