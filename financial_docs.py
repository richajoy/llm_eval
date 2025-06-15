"""
Sample financial regulatory documents for vector database demonstration.
"""

# Sample regulatory documents related to 401(k) plans and investments
FINANCIAL_REGULATIONS = [
    """
    401(k) Early Withdrawal Penalties:
    The IRS generally imposes a 10% early withdrawal penalty on distributions from a 401(k) plan before age 59½.
    Exceptions to this penalty include:
    - Distributions due to total and permanent disability
    - Distributions to beneficiaries after the account holder's death
    - Distributions made as part of a series of substantially equal periodic payments
    - Distributions due to an IRS levy on the plan
    - Distributions that qualify as hardship distributions
    - Distributions after separation from service if the separation occurred during or after the calendar year in which the participant reached age 55
    Even when exceptions apply, income tax on the distribution is still generally required.
    """,
    
    """
    Retirement Investment Regulations:
    The Employee Retirement Income Security Act (ERISA) sets minimum standards for most voluntarily established retirement plans.
    Key provisions include:
    - Fiduciary responsibilities for plan administrators
    - Reporting and disclosure requirements to plan participants
    - Vesting requirements for employer contributions
    - Funding rules to ensure plan solvency
    - Anti-discrimination provisions to ensure plans don't favor highly compensated employees
    - Right of plan participants to sue for benefits and breaches of fiduciary duty
    ERISA is enforced by the Department of Labor, the IRS, and the Pension Benefit Guaranty Corporation.
    """,
    
    """
    IRA vs 401(k) Regulatory Differences:
    While both IRAs and 401(k)s offer tax advantages for retirement savings, they have different regulatory frameworks:
    - Contribution limits: 401(k)s typically have higher contribution limits than IRAs
    - Required Minimum Distributions (RMDs): Both traditional IRAs and 401(k)s require distributions beginning at age 72
    - Early withdrawal penalties: Both generally impose 10% penalties on withdrawals before age 59½
    - Employer involvement: 401(k)s require employer sponsorship, while IRAs are individual accounts
    - Investment options: 401(k)s offer limited options selected by the plan administrator, while IRAs typically provide a broader range of investment options
    - Loan provisions: Many 401(k)s allow loans, while IRAs do not
    Different regulations may apply to Roth versions of these accounts.
    """,
    
    """
    2025 Retirement Contribution Limits:
    For 2025, the IRS has established the following contribution limits for retirement accounts:
    
    401(k), 403(b), and 457 plans:
    - Regular contribution limit: $23,000 for employees under age 50
    - Catch-up contribution limit: Additional $7,500 for employees age 50 and over
    - Total maximum contribution for age 50+: $30,500
    - Combined employer/employee contribution limit: $69,000 or 100% of compensation, whichever is less
    
    Traditional and Roth IRAs:
    - Regular contribution limit: $7,000 for individuals under age 50
    - Catch-up contribution limit: Additional $1,000 for individuals age 50 and over
    - Total maximum contribution for age 50+: $8,000
    
    These limits are subject to income restrictions for certain accounts and may be adjusted annually by the IRS for inflation.
    """
]

# Function to get the documents
def get_financial_docs():
    """Returns the sample financial regulatory documents."""
    return FINANCIAL_REGULATIONS
