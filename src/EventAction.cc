#include "EventAction.hh"

#include "RunAction.hh"

#include "G4Event.hh"
#include "G4SystemOfUnits.hh"

namespace B1
{

EventAction::EventAction(RunAction* runAction)
: G4UserEventAction(),
  fRunAction(runAction),
  fDetectorEdep(0.),
  fDetectorGammaEntries(0),
  fPrimaryGammaEntries(0),
  fTransmissionGammaEntries(0),
  fTransmissionPrimaryGammaEntries(0),
  fSideScatterGammaEntries(0),
  fSideScatterPrimaryGammaEntries(0)
{}

void EventAction::BeginOfEventAction(const G4Event*)
{
  fDetectorEdep = 0.;
  fDetectorGammaEntries = 0;
  fPrimaryGammaEntries = 0;
  fTransmissionGammaEntries = 0;
  fTransmissionPrimaryGammaEntries = 0;
  fSideScatterGammaEntries = 0;
  fSideScatterPrimaryGammaEntries = 0;
}

void EventAction::EndOfEventAction(const G4Event* event)
{
  fRunAction->AddRunDetectorEdep(fDetectorEdep);
  fRunAction->AddRunDetectorGammaEntries(fDetectorGammaEntries);
  fRunAction->AddRunPrimaryGammaEntries(fPrimaryGammaEntries);

  fRunAction->WriteEventData(
    event->GetEventID(),
    fDetectorEdep / keV,
    fDetectorGammaEntries,
    fPrimaryGammaEntries,
    fTransmissionGammaEntries,
    fTransmissionPrimaryGammaEntries,
    fSideScatterGammaEntries,
    fSideScatterPrimaryGammaEntries);
}

void EventAction::AddDetectorGammaEntry(const std::string& detectorId)
{
  ++fDetectorGammaEntries;
  if (detectorId == "side_scatter") {
    ++fSideScatterGammaEntries;
  } else {
    ++fTransmissionGammaEntries;
  }
}

void EventAction::AddPrimaryGammaEntry(const std::string& detectorId)
{
  ++fPrimaryGammaEntries;
  if (detectorId == "side_scatter") {
    ++fSideScatterPrimaryGammaEntries;
  } else {
    ++fTransmissionPrimaryGammaEntries;
  }
}

void EventAction::RecordDetectorHit(G4int eventID,
                                    const std::string& detectorId,
                                    G4double x_mm,
                                    G4double y_mm,
                                    G4double z_mm,
                                    G4double photonEnergy_keV,
                                    G4bool isPrimary,
                                    G4double theta_deg,
                                    G4bool isDirectPrimary,
                                    G4bool isScatteredPrimary)
{
  fRunAction->WriteHitData(
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
