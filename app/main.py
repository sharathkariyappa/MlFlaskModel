# Install dependencies first:
# pip install flask pandas joblib scikit-learn

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import math

app = Flask(__name__)

# Load model and label encoder
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models")

model = joblib.load(os.path.join(MODEL_PATH, 'role_classifier.pkl'))
label_encoder = joblib.load(os.path.join(MODEL_PATH, 'label_encoder.pkl'))


# GitHub scoring function
def calculate_github_score(totalContributions, pullRequests, issues, repositoriesContributedTo, followers, repositories):
    activityScore = 0
    if totalContributions > 0:
        if totalContributions < 100:
            activityScore = totalContributions * 0.1
        elif totalContributions < 500:
            activityScore = 10 + (totalContributions - 100) * 0.05
        elif totalContributions < 2000:
            activityScore = 30 + (totalContributions - 500) * 0.02
        else:
            activityScore = 60 + math.log10(totalContributions / 2000) * 15
    activityScore = min(activityScore, 80)

    prScore = min(pullRequests * 1.5, 40)
    issueScore = min(issues * 1, 20)
    impactScore = prScore + issueScore

    followerScore = min(followers * 0.05, 15)


    repoScore = min(repositories * 2, 15)
    contribScore = min(repositoriesContributedTo * 0.5, 15)
    collaborationScore = repoScore + contribScore

    totalGithubScore = activityScore + impactScore + collaborationScore + followerScore
    return round(totalGithubScore)

# Onchain scoring function
def calculate_onchain_score(ethBalance, txCount, contractDeployments, tokenBalances, nftCount, daoVotes):
    actualEthBalance = ethBalance if ethBalance < 1e6 else ethBalance / 1e18

    wealthScore = 0
    if actualEthBalance > 0:
        if actualEthBalance < 1:
            wealthScore += actualEthBalance * 10
        elif actualEthBalance < 10:
            wealthScore += 10 + (actualEthBalance - 1) * 5
        elif actualEthBalance < 100:
            wealthScore += 55 + (actualEthBalance - 10) * 2
        else:
            wealthScore += 235 + math.log10(actualEthBalance / 100) * 30

    wealthScore = min(wealthScore, 200)

    activityScore = 0
    if txCount > 0:
        if txCount < 100:
            activityScore = txCount * 0.2
        elif txCount < 1000:
            activityScore = 20 + (txCount - 100) * 0.05
        else:
            activityScore = 65 + math.log10(txCount / 1000) * 15
    activityScore = min(activityScore, 80)

    technicalScore = 0
    if contractDeployments > 0:
        technicalScore += 40 + min(contractDeployments * 10, 30)
    if nftCount > 0:
        technicalScore += 10

    governanceScore = min(daoVotes * 2, 30)

    totalOnchainScore = wealthScore + activityScore + technicalScore + governanceScore
    return round(totalOnchainScore)

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Convert booleans
    data['isContractDeployer'] = int(data['isContractDeployer'])
    data['hasNFTs'] = int(data['hasNFTs'])

    # Model prediction
    input_features = [
        data['totalContributions'],
        data['pullRequests'],
        data['issues'],
        data['repositoriesContributedTo'],
        data['followers'],
        data['repositories'],
        data['ethBalance'],
        data['txCount'],
        data['isContractDeployer'],
        data['contractDeployments'],
        data['tokenBalances'],
        data['nftCount'],
        data['daoVotes'],
        data['hasNFTs']
    ]

    prediction = model.predict([input_features])
    predicted_role = label_encoder.inverse_transform(prediction)[0]

    # Scoring
    github_score = calculate_github_score(
        data['totalContributions'],
        data['pullRequests'],
        data['issues'],
        data['repositoriesContributedTo'],
        data['followers'],
        data['repositories'],
    )

    onchain_score = calculate_onchain_score(
        data['ethBalance'],
        data['txCount'],
        data['contractDeployments'],
        data['tokenBalances'],
        data['nftCount'],
        data['daoVotes']
    )

    return jsonify({
        'role': predicted_role,
        'github_score': github_score,
        'onchain_score': onchain_score
    })

if __name__ == '__main__':
    app.run(port=5001)
