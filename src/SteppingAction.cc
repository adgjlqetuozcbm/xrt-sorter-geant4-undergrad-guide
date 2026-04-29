#include "SteppingAction.hh"

#include "DetectorConstruction.hh"
#include "EventAction.hh"
#include "ExperimentConfig.hh"

#include "G4EventManager.hh"
#include "G4Gamma.hh"
#include "G4LogicalVolume.hh"
#include "G4Step.hh"
#include "G4StepPoint.hh"
#include "G4StepStatus.hh"
#include "G4SystemOfUnits.hh"
#include "G4ThreeVector.hh"
#include "G4Track.hh"
#include "G4TouchableHandle.hh"
#include "G4VPhysicalVolume.hh"

namespace B1
{

SteppingAction::SteppingAction(const DetectorConstruction* detectorConstruction,
                               EventAction* eventAction)
: G4UserSteppingAction(),
  fDetConstruction(detectorConstruction),
  fEventAction(eventAction),
  fScoringVolume(nullptr)
{}

void SteppingAction::UserSteppingAction(const G4Step* step)
{
  if (!fScoringVolume) {
    fScoringVolume = fDetConstruction->GetScoringVolume();
  }

  auto preVolume = step->GetPreStepPoint()->GetTouchableHandle()->GetVolume();
  if (!preVolume) return;

  auto preLV = preVolume->GetLogicalVolume();

  auto postPoint = step->GetPostStepPoint();
  auto postVolume = postPoint->GetTouchableHandle()->GetVolume();
  auto postLV = postVolume ? postVolume->GetLogicalVolume() : nullptr;

  // 1) detector energy deposition
  if (fDetConstruction->IsDetectorVolume(preLV)) {
    G4double edep = step->GetTotalEnergyDeposit();
    if (edep > 0.) {
      fEventAction->AddDetectorEdep(edep);
    }
  }

  // 2) detector hit record
  auto track = step->GetTrack();
  if (track->GetDefinition() == G4Gamma::GammaDefinition() &&
      fDetConstruction->IsDetectorVolume(postLV) &&
      postPoint->GetStepStatus() == fGeomBoundary) {

    const auto detectorId = fDetConstruction->DetectorId(postLV);
    fEventAction->AddDetectorGammaEntry(detectorId);

    auto currentEvent = G4EventManager::GetEventManager()->GetConstCurrentEvent();
    G4int eventID = currentEvent ? currentEvent->GetEventID() : -1;

    auto pos = postPoint->GetPosition();
    G4double x_mm = pos.x() / mm;
    G4double y_mm = pos.y() / mm;
    G4double z_mm = pos.z() / mm;
    G4double photonEnergy_keV = postPoint->GetKineticEnergy() / keV;

    G4bool isPrimary = false;
    G4bool isDirectPrimary = false;
    G4bool isScatteredPrimary = false;
    G4double theta_deg = -1.0;

    if (track->GetTrackID() == 1 && track->GetParentID() == 0) {
      isPrimary = true;
      fEventAction->AddPrimaryGammaEntry(detectorId);

      auto dir = track->GetMomentumDirection().unit();
      const auto& config = GetExperimentConfig();
      auto beamDir = G4ThreeVector(config.dirX, config.dirY, config.dirZ).unit();
      G4double theta = dir.angle(beamDir);  // radians
      theta_deg = theta / deg;

      // 第一版工程判据：
      // 若偏转角小于 1 度，则视为近似直透；
      // 否则视为散射后透射。
      if (theta_deg < 1.0) {
        isDirectPrimary = true;
      } else {
        isScatteredPrimary = true;
      }
    }

    fEventAction->RecordDetectorHit(
      eventID,
      detectorId,
      x_mm,
      y_mm,
      z_mm,
      photonEnergy_keV,
      isPrimary,
      theta_deg,
      isDirectPrimary,
      isScatteredPrimary);
  }
}

}
