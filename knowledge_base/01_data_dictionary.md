# Data Dictionary — Freddie Mac Single-Family Loan-Level Dataset

Ce document décrit chaque variable utilisée dans le modèle de scoring crédit, sa source et son interprétation métier.

## Variables d'origination (connues à l'octroi du prêt)

### credit_score
Score FICO de l'emprunteur au moment de l'octroi. Échelle 300–850. Plus le score est élevé, plus l'emprunteur est solvable. C'est historiquement le meilleur prédicteur de défaut. Valeur sentinelle 9999 = manquant. Un FICO inférieur à 660 est considéré comme subprime.

### dti (Debt-to-Income ratio)
Ratio dette/revenu de l'emprunteur, en pourcentage. Mesure la part du revenu mensuel consacrée au remboursement de toutes les dettes. Un DTI supérieur à 43% dépasse le seuil "Qualified Mortgage" (QM) défini par le CFPB et signale un risque accru. Valeur sentinelle 999 = manquant.

### oltv (Original Loan-to-Value)
Ratio prêt/valeur du bien au moment de l'octroi, en pourcentage. Mesure l'apport personnel : un OLTV de 80 signifie que le prêt couvre 80% de la valeur du bien (20% d'apport). Au-delà de 80%, une assurance hypothécaire (MIP) est généralement obligatoire. Valeur sentinelle 999 = manquant.

### ocltv (Original Combined Loan-to-Value)
Comme OLTV mais inclut les prêts juniors (second lien). Toujours supérieur ou égal à OLTV. Plus complet pour mesurer l'endettement total garanti par le bien. Le modèle utilise OCLTV et a retiré OLTV (corrélation 0.99 entre les deux).

### original_upb (Unpaid Principal Balance)
Montant initial du prêt en dollars. Capital restant dû à l'octroi.

### original_interest_rate
Taux d'intérêt nominal du prêt à l'octroi, en pourcentage. Corrélé négativement au FICO : les meilleurs profils obtiennent les meilleurs taux.

### original_loan_term
Durée du prêt en mois. Les valeurs standards sont 360 (30 ans), 180 (15 ans), 240 (20 ans).

### channel
Canal d'origination du prêt. R = Retail (banque directe), C = Correspondent, B = Broker, T = TPO (Third Party Origination).

### occupancy_status
Statut d'occupation du bien. P = résidence principale (Primary), S = résidence secondaire (Secondary), I = investissement (Investment).

### property_type
Type de bien. SF = maison individuelle (Single Family), CO = condominium, PU = Planned Unit Development, MH = mobil-home (Manufactured Housing), CP = coopérative.

### loan_purpose
Motif du prêt. P = achat (Purchase), C = refinancement avec retrait de cash (Cash-out refinance), N = refinancement sans retrait (No cash-out).

### first_time_homebuyer_flag
Indicateur primo-accédant. Y = oui, N = non. Les primo-accédants présentent un risque de défaut plus élevé (7.5% vs 5% dans l'échantillon).

### number_of_borrowers
Nombre de co-emprunteurs sur le prêt. Contre-intuitivement, 2 emprunteurs réduit le risque (filet de sécurité), tandis qu'un seul ou trois et plus l'augmente.

### property_state
État américain où se situe le bien (code à 2 lettres). Encode un fort signal géographique : NY, HI, LA, FL, CT affichent les taux de défaut les plus élevés.

### msa (Metropolitan Statistical Area)
Code de la zone métropolitaine. Granularité géographique intermédiaire entre l'état et le code postal. Manquant pour les zones rurales.

## Variables de performance (utilisées uniquement pour construire la cible)

### current_loan_delinquency_status
Nombre de mois de retard de paiement à une date donnée. 0 = à jour, 1 = 30 jours de retard, 2 = 60 jours, 3 = 90 jours et plus.

### zero_balance_code
Code expliquant pourquoi le solde du prêt est tombé à zéro. 01 = remboursé volontairement (prepaid), 03 = saisie (foreclosure), 09 = REO (bien repris par le prêteur).

## Variable cible

### default
Variable binaire construite pour la modélisation. default = 1 si le prêt a atteint 90+ jours de retard (delinquency >= 3) OU a fini en saisie/REO (zero_balance_code dans {03, 09}). Sinon default = 0. Taux de défaut observé dans l'échantillon : 5.57%.

## Features dérivées (feature engineering)

### is_subprime
Flag = 1 si credit_score < 660.

### is_high_ltv
Flag = 1 si ocltv > 95.

### is_high_dti
Flag = 1 si dti > 43.

### risk_count
Somme des trois flags ci-dessus (0 à 3). Le taux de défaut monte de 4% (risk_count=0) à 20% (risk_count=3).

### fico_dti_interaction
Produit (850 - credit_score) × dti / 100. Capture les profils cumulant un FICO faible et un DTI élevé. C'est le driver #1 du modèle selon l'analyse SHAP.

### fico_ltv_interaction
Produit (850 - credit_score) × ocltv / 100. Capture les profils cumulant FICO faible et fort endettement sur le bien.

### monthly_payment
Mensualité approximée par la formule du prêt à amortissement constant.

### rate_spread
Écart entre le taux du prêt et le taux moyen de son millésime d'origination. Proxy du scoring interne appliqué par le vendeur du prêt.
