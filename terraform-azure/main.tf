# ================================================================
# TERRAFORM - DÉPLOIEMENT AZURE CONTAINER APPS
# ================================================================

terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.90"
    }
  }
}

# ================================================================
# PROVIDER AZURE
# ================================================================

provider "azurerm" {
  features {}
}

# ================================================================
# VARIABLES
# ================================================================

variable "nom_projet" {
  description = "Préfixe utilisé pour nommer les ressources Azure"
  type        = string
  default     = "agri-anomaly-beja"
}

variable "region" {
  description = "Région Azure"
  type        = string
  default     = "francecentral"
}

variable "image_api" {
  description = "Image Docker de l'API"
  type        = string
  default     = "ghcr.io/manelkid/agri-anomaly-api:latest"
}

variable "image_frontend" {
  description = "Image Docker du frontend"
  type        = string
  default     = "ghcr.io/manelkid/agri-anomaly-frontend:latest"
}

variable "ghcr_username" {
  description = "Nom d'utilisateur GitHub pour GHCR. Laisser vide si les images sont publiques."
  type        = string
  default     = ""
}

variable "ghcr_token" {
  description = "Personal Access Token GitHub avec le scope read:packages."
  type        = string
  default     = ""
  sensitive   = true
}

locals {
  ghcr_auth_enabled = var.ghcr_username != "" && var.ghcr_token != ""
}

# ================================================================
# RESSOURCE 1 : RESOURCE GROUP
# ================================================================

resource "azurerm_resource_group" "principal" {
  name     = "${var.nom_projet}-rg"
  location = var.region
}

# ================================================================
# RESSOURCE 2 : LOG ANALYTICS
# ================================================================

resource "azurerm_log_analytics_workspace" "logs" {
  name                = "${var.nom_projet}-logs"
  location            = azurerm_resource_group.principal.location
  resource_group_name = azurerm_resource_group.principal.name

  sku               = "PerGB2018"
  retention_in_days = 30
}

# ================================================================
# RESSOURCE 3 : CONTAINER APPS ENVIRONMENT
# ================================================================

resource "azurerm_container_app_environment" "environnement" {
  name                = "${var.nom_projet}-env"
  location            = azurerm_resource_group.principal.location
  resource_group_name = azurerm_resource_group.principal.name

  log_analytics_workspace_id = azurerm_log_analytics_workspace.logs.id
}

# ================================================================
# RESSOURCE 4 : API
# ================================================================

resource "azurerm_container_app" "api" {
  name                         = "${var.nom_projet}-api"
  container_app_environment_id = azurerm_container_app_environment.environnement.id
  resource_group_name          = azurerm_resource_group.principal.name

  revision_mode = "Single"

  # --------------------------------------------------------------
  # AUTHENTIFICATION GHCR
  # --------------------------------------------------------------

  dynamic "secret" {
    for_each = local.ghcr_auth_enabled ? [1] : []

    content {
      name  = "ghcr-pat"
      value = var.ghcr_token
    }
  }

  dynamic "registry" {
    for_each = local.ghcr_auth_enabled ? [1] : []

    content {
      server               = "ghcr.io"
      username             = var.ghcr_username
      password_secret_name = "ghcr-pat"
    }
  }

  # --------------------------------------------------------------
  # CONTENEUR
  # --------------------------------------------------------------

  template {
    min_replicas = 1
    max_replicas = 3

    container {
      name   = "api"
      image  = var.image_api
      cpu    = 0.5
      memory = "1Gi"
    }
  }

  # --------------------------------------------------------------
  # INGRESS
  # --------------------------------------------------------------

  ingress {
    external_enabled = true
    target_port      = 8000

    traffic_weight {
      percentage      = 100
      latest_revision = true
    }
  }
}

# ================================================================
# RESSOURCE 5 : FRONTEND
# ================================================================

resource "azurerm_container_app" "frontend" {
  name                         = "${var.nom_projet}-frontend"
  container_app_environment_id = azurerm_container_app_environment.environnement.id
  resource_group_name          = azurerm_resource_group.principal.name

  revision_mode = "Single"

  # --------------------------------------------------------------
  # AUTHENTIFICATION GHCR
  # --------------------------------------------------------------

  dynamic "secret" {
    for_each = local.ghcr_auth_enabled ? [1] : []

    content {
      name  = "ghcr-pat"
      value = var.ghcr_token
    }
  }

  dynamic "registry" {
    for_each = local.ghcr_auth_enabled ? [1] : []

    content {
      server               = "ghcr.io"
      username             = var.ghcr_username
      password_secret_name = "ghcr-pat"
    }
  }

  # --------------------------------------------------------------
  # CONTENEUR
  # --------------------------------------------------------------

  template {
    min_replicas = 1
    max_replicas = 3

    container {
      name   = "frontend"
      image  = var.image_frontend
      cpu    = 0.25
      memory = "0.5Gi"
    }
  }

  # --------------------------------------------------------------
  # INGRESS
  # --------------------------------------------------------------

  ingress {
    external_enabled = true
    target_port      = 8080

    traffic_weight {
      percentage      = 100
      latest_revision = true
    }
  }
}

# ================================================================
# OUTPUT 1 : URL FRONTEND
# ================================================================

output "url_frontend" {
  description = "URL publique du frontend"

  value = "https://${azurerm_container_app.frontend.latest_revision_fqdn}"
}

# ================================================================
# OUTPUT 2 : URL API
# ================================================================

output "url_api" {
  description = "URL publique de l'API"

  value = "https://${azurerm_container_app.api.latest_revision_fqdn}"
}

# ================================================================
# OUTPUT 3 : SWAGGER
# ================================================================

output "url_swagger" {
  description = "Documentation Swagger de l'API"

  value = "https://${azurerm_container_app.api.latest_revision_fqdn}/docs"
}