TARGET_COLS = 'SalePrice'
ID_COL = 'Id'
CAT_COLS = [
'MSZoning',
'Street',
'Alley',
'LotShape',
'LandContour',
'Utilities',
'LotConfig',
'LandSlope',
'Neighborhood',
'Condition1',
'Condition2',
'BldgType',
'HouseStyle',
'RoofStyle',
'RoofMatl',
'Exterior1st',
'Exterior2nd',
'MasVnrType',
'ExterQual',
'ExterCond',
'Foundation',
'BsmtQual',
'BsmtCond',
'BsmtExposure',
'BsmtFinType1',
'BsmtFinType2',
'Heating',
'HeatingQC',
'CentralAir',
'Electrical',
'KitchenQual',
'Functional',
'FireplaceQu',
'GarageType',
'GarageFinish',
'GarageQual',
'GarageCond',
'PavedDrive',
'PoolQC',
'Fence',
'MiscFeature',
'SaleType',
'SaleCondition']
REAL_COLS = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']
NAN_COLS = ['LotFrontage',
'Alley',
'MasVnrType',
'MasVnrArea',
'BsmtQual',
'BsmtCond',
'BsmtExposure',
'BsmtFinType1',
'BsmtFinType2',
'Electrical',
'FireplaceQu',
'GarageType',
'GarageYrBlt',
'GarageFinish',
'GarageQual',
'GarageCond',
'PoolQC',
'Fence',
'MiscFeature',
'MSZoning']

FLOAT_NAN_COLS = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']

ZONING_COL = 'MSZoning'

OHE_COLS = ['Is_RL', 'Is_RH', 'Is_RM', 'Is_C (all)', 'Is_FV']

