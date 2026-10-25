"""Canonical merchants/counterparties with their aliases.

Used both by the synthetic generator (to render descriptions) and by merchant
normalisation (to map noisy descriptions back to a canonical name). Names are real
Ethiopian brands/institutions as they would appear on a statement; all transactions
generated from them are synthetic.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Merchant:
    name: str
    category: str
    aliases: tuple[str, ...]
    region: str = "ET"  # "ET" = Ethiopian synthetic, "US" = seen in the Kaggle export


MERCHANTS: tuple[Merchant, ...] = (
    # Telecom & airtime
    Merchant(
        "Ethio Telecom",
        "Telecom & Airtime",
        ("ethio telecom", "ethiotelecom", "ethio tele", "ኢትዮ ቴሌኮም"),
    ),
    Merchant(
        "telebirr Airtime",
        "Telecom & Airtime",
        ("telebirr airtime", "telebirr recharge", "telebirr package", "ቴሌብር የአየር ሰዓት"),
    ),
    Merchant(
        "Safaricom Ethiopia",
        "Telecom & Airtime",
        ("safaricom", "safaricom et", "m-pesa airtime", "ሳፋሪኮም"),
    ),
    # Transfers (channel / bank as counterparty)
    Merchant(
        "Commercial Bank of Ethiopia",
        "Transfers",
        ("cbe", "cbe birr", "commercial bank of ethiopia", "ንግድ ባንክ"),
    ),
    Merchant(
        "telebirr Transfer",
        "Transfers",
        ("telebirr send money", "telebirr transfer", "telebirr p2p", "ቴሌብር ገንዘብ"),
    ),
    Merchant("Awash Bank", "Transfers", ("awash bank", "awash", "አዋሽ ባንክ")),
    Merchant("Dashen Bank", "Transfers", ("dashen bank", "amole", "ዳሽን ባንክ")),
    Merchant("Bank of Abyssinia", "Transfers", ("bank of abyssinia", "boa", "አቢሲኒያ ባንክ")),
    # Cash
    Merchant("ATM Withdrawal", "Cash Withdrawal", ("atm wdl", "atm withdrawal", "ኤቲኤም")),
    # Loans
    Merchant(
        "CBE Loan", "Loan & Card Payments", ("cbe loan repayment", "loan instalment", "ብድር ክፍያ")
    ),
    Merchant(
        "Condominium Loan",
        "Loan & Card Payments",
        ("condominium loan", "condo loan", "ኮንዶሚኒየም ብድር"),
    ),
    # Housing
    Merchant("House Rent", "Housing", ("house rent", "rent payment", "ቤት ኪራይ")),
    # Utilities
    Merchant(
        "Ethiopian Electric Utility",
        "Utilities",
        ("eeu", "ethiopian electric utility", "eeu prepaid", "የመብራት ክፍያ"),
    ),
    Merchant("AAWSA Water", "Utilities", ("aawsa", "aawsa water", "water bill", "የውሃ ክፍያ")),
    # Groceries
    Merchant("Shoa Supermarket", "Groceries", ("shoa supermarket", "shoa", "ሸዋ ሱፐርማርኬት")),
    Merchant("Queens Supermarket", "Groceries", ("queens supermarket", "queens", "ኩዊንስ")),
    Merchant("Safeway Supermarket", "Groceries", ("safeway supermarket", "safeway")),
    Merchant("Fantu Supermarket", "Groceries", ("fantu supermarket", "fantu")),
    Merchant("Atkilt Tera Market", "Groceries", ("atkilt tera", "gulit market", "አትክልት ተራ")),
    # Dining & coffee
    Merchant("Tomoca Coffee", "Dining & Coffee", ("tomoca", "tomoca coffee", "ቶሞካ")),
    Merchant("Kaldis Coffee", "Dining & Coffee", ("kaldis", "kaldis coffee", "ካልዲስ")),
    Merchant("Yod Abyssinia", "Dining & Coffee", ("yod abyssinia", "yod abyssinia restaurant")),
    Merchant("Kategna Restaurant", "Dining & Coffee", ("kategna", "kategna restaurant", "ቃተኛ")),
    Merchant("Effoi Pizza", "Dining & Coffee", ("effoi pizza", "effoi")),
    # Transport & fuel
    Merchant("Ride Taxi", "Transport & Fuel", ("ride taxi", "ride ethiopia", "ራይድ")),
    Merchant("Feres", "Transport & Fuel", ("feres", "feres ride", "ፈረስ")),
    Merchant(
        "TotalEnergies",
        "Transport & Fuel",
        ("totalenergies", "total energies", "total fuel", "ቶታል"),
    ),
    Merchant("NOC Fuel", "Transport & Fuel", ("noc fuel", "noc station", "ኖክ ነዳጅ")),
    # Shopping
    Merchant("Edna Mall", "Shopping", ("edna mall", "ednamall", "ኤድና ሞል")),
    Merchant("Friendship City Center", "Shopping", ("friendship city center", "friendship mall")),
    Merchant("Merkato", "Shopping", ("merkato", "mercato", "መርካቶ")),
    Merchant("Jiji Ethiopia", "Shopping", ("jiji", "jiji ethiopia")),
    # Entertainment & subscriptions
    Merchant(
        "DStv",
        "Entertainment & Subscriptions",
        ("dstv", "dstv subscription", "multichoice", "ዲኤስቲቪ"),
    ),
    Merchant("Canal+ Ethiopia", "Entertainment & Subscriptions", ("canal+", "canal plus")),
    # Health & personal care
    Merchant(
        "Hallelujah Hospital",
        "Health & Personal Care",
        ("hallelujah hospital", "hallelujah gen hospital"),
    ),
    Merchant(
        "Kenema Pharmacy", "Health & Personal Care", ("kenema pharmacy", "kenema", "ቀነማ ፋርማሲ")
    ),
    Merchant(
        "St. Paul Hospital",
        "Health & Personal Care",
        ("st paul hospital", "st. paul hospital", "ቅዱስ ጳውሎስ ሆስፒታል"),
    ),
    # Insurance
    Merchant("Nyala Insurance", "Insurance", ("nyala insurance", "nyala ins", "ኒያላ ኢንሹራንስ")),
    Merchant(
        "Ethiopian Insurance Corporation",
        "Insurance",
        ("ethiopian insurance corporation", "eic insurance"),
    ),
    # Education
    Merchant("Addis Ababa University", "Education", ("addis ababa university", "aau tuition")),
    Merchant("School Fee", "Education", ("school fee", "school fees", "የትምህርት ቤት ክፍያ")),
    # Travel
    Merchant(
        "Ethiopian Airlines",
        "Travel",
        ("ethiopian airlines", "ethiopian airline", "et airlines", "የኢትዮጵያ አየር መንገድ"),
    ),
    Merchant("Skylight Hotel", "Travel", ("skylight hotel", "skylight")),
    Merchant("Haile Resort", "Travel", ("haile resort", "haile hotel")),
    # Government & fees
    Merchant(
        "Ministry of Revenues",
        "Government & Fees",
        ("ministry of revenues", "mor tax", "erca tax", "ግብር ክፍያ"),
    ),
    Merchant(
        "Immigration (ICS)",
        "Government & Fees",
        ("ics passport", "immigration ics", "passport renewal"),
    ),
    Merchant(
        "Bank Service Charge", "Government & Fees", ("service charge", "sms charge", "የአገልግሎት ክፍያ")
    ),
    # Community & charity
    Merchant("Iddir", "Community & Charity", ("iddir", "edir", "እድር")),
    Merchant(
        "Church Tithe",
        "Community & Charity",
        ("church tithe", "medhanealem church", "አስራት", "ቤተክርስቲያን"),
    ),
    Merchant("Mosque Donation", "Community & Charity", ("mosque donation", "zakat", "መስጊድ")),
    # Savings
    Merchant("Equb", "Savings", ("equb", "ekub", "እቁብ")),
    Merchant("Savings Transfer", "Savings", ("to savings", "savings account", "ቁጠባ")),
    # Merchants seen in the US Kaggle export
    Merchant("Amazon", "Shopping", ("amazon", "amazon mktplace"), "US"),
    Merchant("Target", "Shopping", ("target",), "US"),
    Merchant("Starbucks", "Dining & Coffee", ("starbucks",), "US"),
    Merchant("Netflix", "Entertainment & Subscriptions", ("netflix", "netflix.com"), "US"),
    Merchant("Spotify", "Entertainment & Subscriptions", ("spotify",), "US"),
    Merchant("Amazon Video", "Entertainment & Subscriptions", ("amazon video",), "US"),
    Merchant("Shell", "Transport & Fuel", ("shell", "shell oil"), "US"),
    Merchant("BP", "Transport & Fuel", ("bp",), "US"),
    Merchant("Chevron", "Transport & Fuel", ("chevron",), "US"),
    Merchant("State Farm", "Insurance", ("state farm",), "US"),
    Merchant("Best Buy", "Shopping", ("best buy",), "US"),
    Merchant("Chick-Fil-A", "Dining & Coffee", ("chick-fil-a", "chick fil a"), "US"),
    Merchant("Wendy's", "Dining & Coffee", ("wendy's", "wendys"), "US"),
    Merchant("Bojangles", "Dining & Coffee", ("bojangles",), "US"),
)
