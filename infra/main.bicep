// Databricks Lakehouse MLOps lab — core infrastructure
// Scope: resource group (create the RG first via deploy.sh)
targetScope = 'resourceGroup'

@description('Short suffix to keep names globally unique (lowercase letters/digits).')
param suffix string = toLower(substring(uniqueString(resourceGroup().id), 0, 6))

@description('Azure region for all resources.')
param location string = resourceGroup().location

var storageName = 'stdbxchurn${suffix}'           // ADLS Gen2, max 24 chars
var databricksName = 'dbw-churn-lab-${suffix}'
var accessConnectorName = 'dbac-churn-lab-${suffix}'
var managedRgName = 'rg-dbx-churn-lab-managed-${suffix}'

// ---------- ADLS Gen2 (hierarchical namespace ON) ----------
resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: storageName
  location: location
  sku: { name: 'Standard_LRS' }
  kind: 'StorageV2'
  properties: {
    isHnsEnabled: true                  // required for ADLS Gen2 / Unity Catalog
    minimumTlsVersion: 'TLS1_2'
    supportsHttpsTrafficOnly: true
    allowBlobPublicAccess: false
  }
}

resource blobService 'Microsoft.Storage/storageAccounts/blobServices@2023-05-01' = {
  parent: storage
  name: 'default'
}

// Containers: raw landing zone + one per medallion layer
resource containers 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = [
  for c in ['raw', 'bronze', 'silver', 'gold']: {
    parent: blobService
    name: c
  }
]

// ---------- Databricks workspace (Premium — needed for Unity Catalog / RBAC) ----------
resource databricks 'Microsoft.Databricks/workspaces@2024-05-01' = {
  name: databricksName
  location: location
  sku: { name: 'premium' }
  properties: {
    managedResourceGroupId: subscriptionResourceId(
      'Microsoft.Resources/resourceGroups',
      managedRgName
    )
  }
}

// ---------- Access Connector (managed identity bridge for Unity Catalog) ----------
resource accessConnector 'Microsoft.Databricks/accessConnectors@2024-05-01' = {
  name: accessConnectorName
  location: location
  identity: { type: 'SystemAssigned' }
}

// Grant the connector Storage Blob Data Contributor on the lake
resource lakeRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(storage.id, accessConnector.id, 'blob-contributor')
  scope: storage
  properties: {
    roleDefinitionId: subscriptionResourceId(
      'Microsoft.Authorization/roleDefinitions',
      'ba92f5b4-2d11-453d-a403-e96b0029c9fe' // Storage Blob Data Contributor
    )
    principalId: accessConnector.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

output storageAccountName string = storage.name
output databricksWorkspaceUrl string = databricks.properties.workspaceUrl
output accessConnectorId string = accessConnector.id
