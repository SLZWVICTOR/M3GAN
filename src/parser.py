import argparse

parser = argparse.ArgumentParser(description='Time-Series Anomaly Detection')
parser.add_argument('--dataset', 
					metavar='-d', 
					type=str, 
					required=False,
					default='synthetic',
                    help="dataset from ['synthetic', 'SMD']")
parser.add_argument('--model', 
					metavar='-m', 
					type=str, 
					required=False,
					default='LSTM_Multivariate',
                    help="model name")
parser.add_argument('--test', 
					action='store_true', 
					help="test the model")
parser.add_argument('--retrain', 
					action='store_true', 
					help="retrain the model")
parser.add_argument('--less', 
					action='store_true', 
					help="train using less data")
parser.add_argument('--exefile',
					metavar='-m',
					type=str,
					required=False,
					default='need file',
                    help="exefile")
parser.add_argument('--subset',
					metavar='-m',
					type=str,
					required=False,
					default='need subset',
                    help="subset")
parser.add_argument('--q',
					type=float,
					required=False,
					default='1e-5',
                    help="q")
parser.add_argument('--ws',
					type=float,
					required=False,
					default='5',
                    help="ws")
args = parser.parse_args()