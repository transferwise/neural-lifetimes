from dataclasses import dataclass


@dataclass
class DataModel:
    pass


@dataclass
class EventprofilesDataModel(DataModel):
    cont_feat_events = [
        "INVOICE_VALUE_GBP_log",
        "FEE_INVOICE_ratio",
        "MARGIN_GBP",
    ]
    cont_feat_profiles = ["AGE_YEARS"]

    discr_feat_events = [
        "ACTION_STATE",
        "ACTION_TYPE",
        "BALANCE_STEP_TYPE",
        "BALANCE_TRANSACTION_TYPE",
        "PRODUCT_TYPE",
        "SENDER_TYPE",
        "SOURCE_CURRENCY",
        "SUCCESSFUL_ACTION",
        "TARGET_CURRENCY",
    ]

    discr_feat_profiles = [
        "ADDRESS_MARKET",
        "ADDR_COUNTRY",
        "BEST_GUESS_COUNTRY",
        "CREATION_PLATFORM",
        "CUSTOMER_TYPE",
        "FIRST_CCY_SEND",
        "FIRST_CCY_TARGET",
        "FIRST_VISIT_IP_COUNTRY",
        "LANGUAGE",
    ]

    target_cols = ["dt"]
    cont_feat = cont_feat_events + cont_feat_profiles
    discr_feat = discr_feat_events + discr_feat_profiles


@dataclass
class EventsOnlyDataModel(DataModel):
    cont_feat = [
        "INVOICE_VALUE_GBP",
        "FEE_VALUE_GBP",
        "MARGIN_GBP",
    ]
    cont_feat_profiles = []
    discr_feat = [
        "ACTION_STATE",
        "ACTION_TYPE",
        "BALANCE_STEP_TYPE",
        "BALANCE_TRANSACTION_TYPE",
        "PRODUCT_TYPE",
        "SENDER_TYPE",
        "SOURCE_CURRENCY",
        "SUCCESSFUL_ACTION",
        "TARGET_CURRENCY",
    ]
    discr_feat_profiles = []
    target_cols = ["dt"]
