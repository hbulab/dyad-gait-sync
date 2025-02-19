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


# ---------- Graph values ----------

COLORS_SOC_REL = ["black", "red", "blue", "green", "orange"]
COLORS_INTERACTION = ["blue", "red", "green", "orange"]
