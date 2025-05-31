# ---------- ATC constants ----------
DAYS_ATC = ["0109", "0217", "0424", "0505", "0508"]
SOCIAL_RELATIONS_JP = ["idontknow", "koibito", "doryo", "kazoku", "yuujin"]
SOCIAL_RELATIONS_EN = ["idontknow", "Couples", "Colleagues", "Families", "Friends"]
BOUNDARIES_ATC = {"xmin": -41000, "xmax": 49000, "ymin": -27500, "ymax": 24000}
BOUNDARIES_ATC_CORRIDOR = {"xmin": 5000, "xmax": 48000, "ymin": -27000, "ymax": 8000}


# ---------- DIAMOR constants ----------
DAYS_DIAMOR = ["06", "08"]
INTENSITIES_OF_INTERACTION_NUM = ["0", "1", "2", "3"]
BOUNDARIES_DIAMOR = {"xmin": -200, "xmax": 60300, "ymin": -5300, "ymax": 12000}
BOUNDARIES_DIAMOR_CORRIDOR = {
    "xmin": 20000,
    "xmax": 60300,
    "ymin": -5300,
    "ymax": 12000,
}
BOUNDARIES_DIAMOR_CENTER_CORRIDOR = {
    "xmin": 30000,
    "xmax": 50000,
    "ymin": 0,
    "ymax": 7500,
}

# ---------- General constants ----------
# velocity threshold
VEL_MIN = 500
VEL_MAX = 3000

# group size threshold
GROUP_BREADTH_MIN = 100
GROUP_BREADTH_MAX = 3000

MAX_DELTA_F = 10

# ---------- Graph values ----------

COLORS_SOC_REL = ["black", "red", "blue", "green", "orange"]
COLORS_INTERACTION = ["blue", "red", "green", "orange"]


# ---------- Graph parameters ----------
DYADS_PARAMETERS = {
    "interaction": {
        0: {
            "color": "blue",
            "label": "Interaction 0",
            "marker": "o",
            "short_label": "0",
        },
        1: {
            "color": "red",
            "label": "Interaction 1",
            "marker": "s",
            "short_label": "1",
        },
        2: {
            "color": "green",
            "label": "Interaction 2",
            "marker": "D",
            "short_label": "2",
        },
        3: {
            "color": "orange",
            "label": "Interaction 3",
            "marker": "^",
            "short_label": "3",
        },
        4: {
            "color": "gray",
            "label": "$RP$",
            "marker": None,
            "short_label": "$RP$",
        },
        5: {
            "color": "indigo",
            "label": "$IP$",
            "marker": None,
            "short_label": "$IP$",
        },
    },
    "contact": {
        0: {"color": "blue", "label": "No contact", "marker": "o"},
        1: {"color": "red", "label": "Contact", "marker": "s"},
        2: {"color": "purple", "label": "Baseline", "marker": "x"},
    },
    "individual": {
        "color": "turquoise",
        "label": "Individuals",
        "marker": "x",
    },
    "metrics": {
        "gsi": {"label": "GSI", "limits": [0, 0.8], "table_label": "GSI"},
        "coherence": {"label": "CWC", "limits": [0, 1], "table_label": "CWC"},
        "delta_f": {
            "label": "$\\Delta f$ [Hz]",
            "limits": [0, 0.4],
            "table_label": "difference in stride frequency $\\Delta f$",
        },
        "rec": {
            "label": "\\%REC",
            "limits": [0, 1],
            "table_label": "percentage of recurrence $\\%\\text{REC}$",
        },
        "det": {
            "label": "\\%DET",
            "limits": [0.7, 1],
            "table_label": "percentage of determinism $\\%\\text{DET}$",
        },
        "maxline": {
            "label": "MAXLINE",
            "limits": [0, 1],
            "table_label": "maximal line length $\\text{MAXLINE}$ ",
        },
        "lyapunov": {
            "label": "maximal Lyapunov exponent",
            "limits": [200 * 10 ** (-3), 1],
            "table_label": "maximal Lyapunov exponent $l_{lyap}$",
        },
        "determinism": {
            "label": "Determinism",
            "limits": [0.5, 1],
            "table_label": "determinism $D$",
        },
    },
}

TRIADS_PARAMETERS = {
    "formation": {
        "v": {"color": "#eae2b7", "label": "$\\vee$", "marker": "o"},
        "lambda": {"color": "#fcbf49", "label": "$\\wedge$", "marker": "s"},
        "abreast": {
            "color": "#2a9d8f",
            "label": "$\\longleftrightarrow$",
            "marker": "^",
        },
        "following": {
            "color": "#264653",
            "label": "$\\updownarrow$",
            "marker": "v",
        },
    },
    "pair": {
        "left_center": {"color": "#eae2b7", "label": "L--C", "marker": "o"},
        "right_center": {
            "color": "#fcbf49",
            "label": "R--C",
            "marker": "s",
        },
        "left_right": {"color": "#2a9d8f", "label": "L--R", "marker": "^"},
        "front_center": {
            "color": "#264653",
            "label": "Front-Center",
            "marker": "v",
        },
        "front_back": {"color": "#e76f51", "label": "Front-Back", "marker": "x"},
        "back_center": {"color": "#f4a261", "label": "Front-Left", "marker": "D"},
    },
    "positions": {
        "left": {"color": "#eae2b7", "label": "Left", "marker": "o"},
        "center": {"color": "#fcbf49", "label": "Center", "marker": "s"},
        "right": {"color": "#2a9d8f", "label": "Right", "marker": "^"},
        "front": {"color": "#264653", "label": "Front", "marker": "v"},
        "back": {"color": "#e76f51", "label": "Back", "marker": "x"},
    },
    "metrics": {
        "gsi": {"label": "GSI", "limits": [0, 0.2]},
        "coherence": {"label": "CWC", "limits": [0, 0.45]},
        "delta_f": {"label": "$\\Delta f$ (Hz)", "limits": [0, 0.2]},
        "lyapunov": {
            "label": "$l_{lyap}$ ($\\times 10^{-3}$)",
            "limits": [200 * 10 ** (-3), 0.1],
        },
        "determinism": {"label": "$D$", "limits": [0.5, 0.5]},
        "rec": {"label": "\%REC", "limits": [0.1, 0.35]},
        "det": {"label": "\%DET", "limits": [0.8, 1]},
        "maxline": {"label": "MAXLINE", "limits": [50, 150]},
    },
}
