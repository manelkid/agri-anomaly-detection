# ================================================================
# TERRAFORM - DÉPLOIEMENT SUR AZURE CONTAINER APPS
# ================================================================
# Prérequis avant "terraform apply" :
#   1. Compte Azure for Students créé (azure.microsoft.com/free/students)
#   2. Azure CLI installé, connecté via : az login
#   3. Image publiée sur ghcr.io (déjà fait via GitHub Actions)

terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.90"
    }
  }
}

provider "azurerm" {
  features {}
}

# --- Variables ---
variable "nom_projet" {
  description = "Préfixe utilisé pour nommer toutes les ressources Azure"
  type        = string
  default     = "agri-anomaly-beja"
}

variable "region" {
  description = "Région Azure (France Central est proche de la Tunisie)"
  type        = string
  default     = "francecentral"
}

variable "image_docker" {
  description = "Image Docker publiée sur GitHub Container Registry"
  type        = string
  default     = "ghcr.io/manelkid/agri-anomaly-api:latest"
}

# --- Ressource 1 : groupe de ressources (dossier qui regroupe tout) ---
resource "azurerm_resource_group" "principal" {
  name     = "${var.nom_projet}-rg"
  location = var.region
}

# --- Ressource 2 : espace de logs (obligatoire pour Container Apps) ---
resource "azurerm_log_analytics_workspace" "logs" {
  name                = "${var.nom_projet}-logs"
  location            = azurerm_resource_group.principal.location
  resource_group_name = azurerm_resource_group.principal.name
  sku                 = "PerGB2018"
  retention_in_days   = 30
}

# --- Ressource 3 : environnement Container Apps ---
resource "azurerm_container_app_environment" "environnement" {
  name                       = "${var.nom_projet}-env"
  location                   = azurerm_resource_group.principal.location
  resource_group_name        = azurerm_resource_group.principal.name
  log_analytics_workspace_id = azurerm_log_analytics_workspace.logs.id
}

# --- Ressource 4 : l'application conteneurisée elle-même ---
resource "azurerm_container_app" "api" {
  name                         = "${var.nom_projet}-api"
  container_app_environment_id = azurerm_container_app_environment.environnement.id
  resource_group_name          = azurerm_resource_group.principal.name
  revision_mode                = "Single"

  template {
    container {
      name   = "api"
      image  = var.image_docker
      cpu    = 0.5
      memory = "1Gi"
    }
  }

  ingress {
    external_enabled = true
    target_port      = 8000
    traffic_weight {
      percentage      = 100
      latest_revision = true
    }
  }
}

# --- Sortie : l'URL publique une fois déployé ---
output "url_publique" {
  value       = "https://${azurerm_container_app.api.ingress[0].fqdn}/docs"
  description = "URL publique de la documentation Swagger de l'API"
}
