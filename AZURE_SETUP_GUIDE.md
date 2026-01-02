# ☁️ Azure Setup Guide

I have updated your workflow to use Azure instead of AWS. Here is how to configure it.

## 1. Required GitHub Secrets

Go to **Settings** → **Secrets and variables** → **Actions** and add these new secrets:

| Secret Name | Description | Example |
|-------------|-------------|---------|
| `ACR_LOGIN_SERVER` | Your Azure Container Registry URL | `myfraudregistry.azurecr.io` |
| `ACR_USERNAME` | Username for ACR | `myfraudregistry` |
| `ACR_PASSWORD` | Password for ACR | `xxxxxxxxxxxx` |
| `MONGO_DB_URL` | MongoDB Connection String | `mongodb+srv://...` |

*(Note: You can remove the AWS_* secrets if you are no longer using them)*

## 2. How to Get These Credentials

1. **Create an Azure Container Registry (ACR)**:
   ```bash
   az acr create --resource-group MyResourceGroup --name MyFraudRegistry --sku Basic
   ```

2. **Enable Admin User** (to get username/password):
   ```bash
   az acr update -n MyFraudRegistry --admin-enabled true
   ```

3. **Get Credentials**:
   ```bash
   # Get Login Server
   az acr show --name MyFraudRegistry --query loginServer

   # Get Password
   az acr credential show --name MyFraudRegistry
   ```

## 3. Workflow Changes Made

- Replaced AWS ECR login with `docker/login-action`
- Replaced `AWS_ECR_REPO_URI` with `${{ secrets.ACR_LOGIN_SERVER }}/fraud-detection`
- Removed AWS credentials configuration

Your CI/CD pipeline is now ready for Azure! 🚀
