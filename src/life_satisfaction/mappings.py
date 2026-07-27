"""Categorical encodings used by the paper analysis workflow."""

CATEGORY_MAPPINGS = {
    "gender": {
        "Woman": 0,
        "Man": 1
    },
    "int_result": {
        "1. Answer": 0
    },
    "A2": {
        "Average": 1,
        "Well": 2,
        "Very well": 3,
        "Poor": 1,
        "Very poor": 0,
        "Don't know": -1
    },
    "B1": {
        "With some difficulty": 1,
        "Without difficulty": 2,
        "With much difficulty": 0,
        "Not at all": 3
    },
    "B2": {
        "With some difficulty": 1,
        "Without difficulty": 2,
        "With much difficulty": 0,
        "Not at all": 3
    },
    "B3": {
        "With some difficulty": 1,
        "Without difficulty": 2,
        "With much difficulty": 0,
        "Not at all": 3
    },
    "B4": {
        "With some difficulty": 1,
        "Without difficulty": 2,
        "With much difficulty": 0,
        "Not at all": 3
    },
    "B5": {
        "With some difficulty": 1,
        "Without difficulty": 2,
        "With much difficulty": 0,
        "Not at all": 3
    },
    "B6": {
        "With some difficulty": 1,
        "Without difficulty": 2,
        "With much difficulty": 0,
        "Not at all": 3
    },
    "B7": {
        "With some difficulty": 1,
        "Without difficulty": 2,
        "With much difficulty": 0,
        "Not at all": 3
    },
    "B8": {
        "With some difficulty": 1,
        "Without difficulty": 2,
        "With much difficulty": 0,
        "Not at all": 3,
        "Don't know": -1
    },
    "B9": {
        "With some difficulty": 1,
        "Without difficulty": 2,
        "With much difficulty": 0,
        "Not at all": 3,
        "Don't know": -1
    },
    "B10": {
        "With some difficulty": 1,
        "Without difficulty": 2,
        "With much difficulty": 0,
        "Not at all": 3,
        "Don't know": -1,
        "Refuse to answer": -1.5
    },
    "B11": {
        "With some difficulty": 1,
        "Without difficulty": 2,
        "With much difficulty": 0,
        "Not at all": 3,
        "Don't know": -1
    },
    "B12": {
        "With some difficulty": 1,
        "Without difficulty": 2,
        "With much difficulty": 0,
        "Not at all": 3,
        "Don't know": -1
    },
    "B13": {
        "With some difficulty": 1,
        "Without difficulty": 2,
        "With much difficulty": 0,
        "Not at all": 3
    },
    "B14": {
        "No": 0,
        "Yes": 1
    },
    "B17_a": {
        "Yes": 1,
        "No": 0
    },
    "C1": {
        "No": 0,
        "Yes": 1,
        "Don't know": -1
    },
    "C4": {
        "No": 2,
        "Yes, one": 1,
        "Don't know": -1,
        "Yes, more": 0
    },
    "D1": {
        "Often": 3,
        "Rarely": 1,
        "Never": 0,
        "Sometimes": 2,
        "Don't know": -1,
        "Always": 4
    },
    "D2": {
        "Often": 3,
        "Rarely": 1,
        "Never": 0,
        "Sometimes": 2,
        "Don't know": -1,
        "Always": 4,
        "Refuse to answer": -1.5
    },
    "D3": {
        "Often": 3,
        "Rarely": 1,
        "Never": 0,
        "Sometimes": 2,
        "Don't know": -1,
        "Always": 4
    },
    "D4": {
        "Often": 3,
        "Rarely": 1,
        "Never": 0,
        "Sometimes": 2,
        "Don't know": -1,
        "Always": 4
    },
    "D5": {
        "Often": 3,
        "Rarely": 1,
        "Never": 0,
        "Sometimes": 2,
        "Don't know": -1,
        "Always": 4
    },
    "D6": {
        "Often": 3,
        "Rarely": 1,
        "Never": 0,
        "Sometimes": 2,
        "Don't know": -1,
        "Always": 4
    },
    "D7": {
        "Often": 3,
        "Rarely": 1,
        "Never": 0,
        "Sometimes": 2,
        "Don't know": -1,
        "Always": 4,
        "Refuse to answer": -1.5
    },
    "D8": {
        "Often": 3,
        "Rarely": 1,
        "Never": 0,
        "Sometimes": 2,
        "Don't know": -1,
        "Always": 4
    },
    "D9": {
        "Often": 3,
        "Rarely": 1,
        "Never": 0,
        "Sometimes": 2,
        "Don't know": -1,
        "Always": 4,
        "Refuse to answer": -1.5
    },
    "D10": {
        "Often": 3,
        "Rarely": 1,
        "Never": 0,
        "Sometimes": 2,
        "Don't know": -1,
        "Always": 4,
        "Refuse to answer": -1.5
    },
    "D11": {
        "Often": 3,
        "Rarely": 1,
        "Never": 0,
        "Sometimes": 2,
        "Don't know": -1,
        "Always": 4,
        "Refuse to answer": -1.5
    },
    "D12": {
        "Often": 3,
        "Rarely": 1,
        "Never": 0,
        "Sometimes": 2,
        "Don't know": -1,
        "Always": 4,
        "Refuse to answer": -1.5
    },
    "D13": {
        "Often": 3,
        "Rarely": 1,
        "Never": 0,
        "Sometimes": 2,
        "Don't know": -1,
        "Always": 4,
        "Refuse to answer": -1.5
    },
    "D14": {
        "Often": 3,
        "Rarely": 1,
        "Never": 0,
        "Sometimes": 2,
        "Don't know": -1,
        "Always": 4,
        "Refuse to answer": -1.5
    },
    "D15": {
        "Often": 3,
        "Rarely": 1,
        "Never": 0,
        "Sometimes": 2,
        "Don't know": -1,
        "Always": 4,
        "Refuse to answer": -1.5
    },
    "D16": {
        "Often": 3,
        "Rarely": 1,
        "Never": 0,
        "Sometimes": 2,
        "Don't know": -1,
        "Always": 4,
        "Refuse to answer": -1.5
    },
    "D17": {
        "Often": 3,
        "Rarely": 1,
        "Never": 0,
        "Sometimes": 2,
        "Don't know": -1,
        "Always": 4,
        "Refuse to answer": -1.5
    },
    "E3": {
        "Often": 3,
        "Rarely": 1,
        "Never": 0,
        "Sometimes": 2,
        "Don't know": -1,
        "Always": 4
    },
    "E5_a": {
        "No": 0,
        "Yes": 1,
        "Don't know": -1
    },
    "E6": {
        "Yes": 1,
        "No": 0,
        "Refuse to answer": -1
    },
    "E12": {
        "Once a week": 6,
        "2-3 times a week": 7,
        "Less than once a month, but several times a year": 4,
        "Never": 0,
        "Almost every day": 8,
        "Less frequently": 1,
        "Once a month": 3,
        "Every day": 9,
        "Once every fortnight": 5,
        "Don't know": -1
    },
    "E17": {
        "Friends/colleagues": 0,
        "Partner/spouse/boy-/girlfriend": 1,
        "I dont share this with anyone": 2,
        "Siblings": 3,
        "Parents": 4,
        "Others": 5,
        "Don't know": -1,
        "Children": 7,
        "Staff": 8,
        "Other family": 9,
        "Refuse to answer": -1
    },
    "education": {
        "Completed secondary school or more (eksamensskole)": 0,
        "Completed compulsory school (folkeskole, 9 years)": 1
    },
    "job": {
        "Holds an ordinary or supported job": 0,
        "Doesn't hold an ordinary or supported job": 1
    },
    "F10": {
        "No": 0,
        "Yes": 1,
        "Don't know": -1
    },
    "F11": {
        "No": 0,
        "Yes": 1,
        "Don't know": -1
    },
    "F15": {
        "9": 9,
        "8": 8,
        "6": 6,
        "7": 7,
        "2": 2,
        "5": 5,
        "1": 1,
        "4": 4,
        "0": 0,
        "3": 3,
        "Don't know": -1,
        "0 very low": 0,
        "Refuse to answer": -1.5
    },
    "G1": {
        "No": 0,
        "Yes": 1,
        "Refuse to answer": -1
    },
    "G6": {
        "Yes, my mom is alive": 1,
        "Yes, my dad is alive": 2,
        "No": 0,
        "Yes, both are alive": 4,
        "Don't Know": -1
    },
    "G7": {
        "No, none of my parents": 0,
        "Yes, one parent": 1,
        "Yes, both parents": 2,
        "Don't Know": -1
    },
    "G10_a": {
        "No": 0,
        "Yes": 1,
        "Don't know": -1
    },
    "G11_a": {
        "Yes": 1,
        "No": 0,
        "Don't know": -1
    },
    "J1": {
        "Less frequently": 1,
        "Daily": 6,
        "Several times a month": 4,
        "Several times a week": 5,
        "Once a week": 3,
        "Never": 0,
        "Once a month": 2,
        "Don't Know": -1
    },
    "J2": {
        "Less frequently": 1,
        "Daily": 6,
        "Several times a month": 4,
        "Several times a week": 5,
        "Once a week": 3,
        "Never": 0,
        "Once a month": 2,
        "Refuse to answer": -1,
        "99": -1
    },
    "J3": {
        "Less frequently": 1,
        "Daily": 6,
        "Several times a month": 4,
        "Several times a week": 5,
        "Once a week": 3,
        "Never": 0,
        "Once a month": 2,
        "Refuse to answer": -1,
        "99": -1
    },
    "J4": {
        "Less frequently": 1,
        "Daily": 6,
        "Several times a month": 4,
        "Several times a week": 5,
        "Once a week": 3,
        "Never": 0,
        "Once a month": 2,
        "Refuse to answer": -1,
        "99": -1
    },
    "J18": {
        "No": 0,
        "Yes": 1,
        "Don't know": -1
    },
    "J18a": {
        "No": 0,
        "Yes": 1,
        "Don't know": -1
    },
    "J8a": {
        "Yes": 1,
        "No": 0,
        "Don't know": -1
    },
    "J9": {
        "Less frequently": 1,
        "Daily": 6,
        "Several times a month": 4,
        "Several times a week": 5,
        "Once a week": 3,
        "Never": 0,
        "Once a month": 2
    },
    "J10": {
        "Less frequently": 1,
        "Daily": 6,
        "Several times a month": 4,
        "Several times a week": 5,
        "Once a week": 3,
        "Never": 0,
        "Once a month": 2,
        "Don't Know": -1
    },
    "J11": {
        "Less frequently": 1,
        "Daily": 6,
        "Several times a month": 4,
        "Several times a week": 5,
        "Once a week": 3,
        "Never": 0,
        "Once a month": 2,
        "Don't Know": -1
    },
    "J12": {
        "Less frequently": 1,
        "Daily": 6,
        "Several times a month": 4,
        "Several times a week": 5,
        "Once a week": 3,
        "Never": 0,
        "Once a month": 2,
        "Don't Know": -1
    },
    "J13": {
        "Less frequently": 1,
        "Daily": 6,
        "Several times a month": 4,
        "Several times a week": 5,
        "Once a week": 3,
        "Never": 0,
        "Once a month": 2,
        "Don't Know": -1
    },
    "J14": {
        "Less frequently": 1,
        "Daily": 6,
        "Several times a month": 4,
        "Several times a week": 5,
        "Once a week": 3,
        "Never": 0,
        "Once a month": 2,
        "Don't Know": -1
    },
    "J15": {
        "Less frequently": 1,
        "Daily": 6,
        "Several times a month": 4,
        "Several times a week": 5,
        "Once a week": 3,
        "Never": 0,
        "Once a month": 2,
        "Don't Know": -1
    },
    "J16_a": {
        "No": 0,
        "Yes": 1,
        "Don't know": -1
    },
    "J20": {
        "Yes": 1,
        "No": 0,
        "Don't know": -1
    },
    "K1_a": {
        "Yes": 1,
        "No": 0
    },
    "K1": {
        "Less frequently": 1,
        "Daily": 6,
        "Several times a month": 4,
        "Several times a week": 5,
        "Once a week": 3,
        "Never": 0,
        "Once a month": 2
    },
    "K2": {
        "Less frequently": 1,
        "Daily": 6,
        "Several times a month": 4,
        "Several times a week": 5,
        "Once a week": 3,
        "Never": 0,
        "Once a month": 2
    },
    "K3": {
        "Less frequently": 1,
        "Daily": 6,
        "Several times a month": 4,
        "Several times a week": 5,
        "Once a week": 3,
        "Never": 0,
        "Once a month": 2
    },
    "K4": {
        "Less frequently": 1,
        "Daily": 6,
        "Several times a month": 4,
        "Several times a week": 5,
        "Once a week": 3,
        "Never": 0,
        "Once a month": 2
    },
    "L8_1": {
        "Withdraw the money immediately DKK 100,000 (EUR 13,407)": 0,
        "Withdraw the money in 12 months DKK 102,000 (EUR 13,675)": 1,
        "Don't know": 2,
        "Refuse to answer": 3
    },
    "M2": {
        "Salary, fee income": 0,
        "Early retirement/retirement pension": 1,
        "Social security": 2,
        "Other": 3,
        "Pension schemes": 4,
        "Self-employment income": 5,
        "Unemployment benefits": 6,
        "Incapacity benefit": 7,
        "Other welfare": 8,
        "Don't Know": 9,
        "Trading bonds, shares and real estate": 10,
        "Interest income": 11,
        "Inheritance": 12,
        "Black money": 13
    },
    "M8": {
        "Good": 3,
        "Average": 2,
        "Bad": 1,
        "Very good": 4,
        "Very bad": 0,
        "Don't Know": -1,
        "Refuse to answer": -1.5
    },
    "N1": {
        "No": 0,
        "Yes": 1,
        "Don't know": -1,
        "Refuse to answer": -1.5
    },
    "N2": {
        "Yes": 1,
        "No": 0,
        "Don't know": -1
    },
    "N5": {
        "Yes": 1,
        "No": 0,
        "Don't know": -1
    },
    "N8": {
        "Yes": 1,
        "No": 0,
        "Don't know": -1
    },
    "N9": {
        "Yes": 1,
        "No": 0,
        "Don't know": -1
    },
    "N12": {
        "No": 0,
        "Yes": 1,
        "Don't know": -1,
        "Refuse to answer": -1.5
    },
    "N13": {
        "No": 0,
        "Yes": 1,
        "Don't know": -1,
        "Refuse to answer": -1.5
    },
    "N16": {
        "Yes": 1,
        "No": 0,
        "Don't know": -1
    },
    "Q1": {
        "No": 0,
        "Yes": 1,
        "Don't know": -1,
        "Refuse to answer": -1.5
    },
    "Q1_a": {
        "No": 0,
        "Yes": 1,
        "Don't know": -1,
        "Refuse to answer": -1.5
    },
    "Q1_b": {
        "No": 0,
        "Yes": 1,
        "Don't know": -1,
        "Refuse to answer": -1.5
    },
    "Q1_e": {
        "No": 0,
        "Yes": 1,
        "Don't know": -1,
        "Refuse to answer": -1.5
    },
    "Q1_f": {
        "No": 0,
        "Yes": 1,
        "Don't know": -1,
        "Refuse to answer": -1.5
    },
    "Q1_g": {
        "No": 0,
        "Yes": 1,
        "Don't know": -1,
        "Refuse to answer": -1.5
    },
    "Q3_a": {
        "No": 0,
        "Yes": 1,
        "Don't know": -1,
        "Refuse to answer": -1.5
    },
    "Q4_a": {
        "Yes": 1,
        "No": 0,
        "Don't know": -1,
        "Refuse to answer": -1.5
    },
    "Q5": {
        "I would consider it for a while, but probably say yes": 0,
        "I would say yes, without hesitation": 1,
        "I would be very much in doubt": 2,
        "I would say no, without hesitation": 3,
        "Don't Know": 4,
        "I would consider it for a while, but probably say no": 5,
        "Refuse to answer": 6
    },
    "Q6": {
        "I would consider it for a while, but probably say yes": 0,
        "I would say yes, without hesitation": 1,
        "I would say no, without hesitation": 2,
        "I would be very much in doubt": 3,
        "I would consider it for a while, but probably say no": 4,
        "Don't Know": 5,
        "Refuse to answer": 6
    },
    "R1": {
        "None": 0,
        "6-10 times": 1,
        "1-2 times": 2,
        "More than 10 times": 3,
        "Don't Know": 4,
        "3-5 times": 5,
        "Refuse to answer": 6
    }
}
