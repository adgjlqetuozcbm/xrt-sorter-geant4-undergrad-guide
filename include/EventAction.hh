#ifndef B1EventAction_h
#define B1EventAction_h 1

#include "G4UserEventAction.hh"
#include "globals.hh"

#include <string>

class G4Event;

namespace B1
{

class RunAction;

class EventAction : public G4UserEventAction
{
  public:
    explicit EventAction(RunAction* runAction);
    ~EventAction() override = default;

    void BeginOfEventAction(const G4Event* event) override;
    void EndOfEventAction(const G4Event* event) override;

    void AddDetectorEdep(G4double edep) { fDetectorEdep += edep; }
    void AddDetectorGammaEntry(const std::string& detectorId);
    void AddPrimaryGammaEntry(const std::string& detectorId);

    void RecordDetectorHit(G4int eventID,
                           const std::string& detectorId,
                           G4double x_mm,
                           G4double y_mm,
                           G4double z_mm,
                           G4double photonEnergy_keV,
                           G4bool isPrimary,
                           G4double theta_deg,
                           G4bool isDirectPrimary,
                           G4bool isScatteredPrimary);

  private:
    RunAction* fRunAction;

    G4double fDetectorEdep;
    G4int fDetectorGammaEntries;
    G4int fPrimaryGammaEntries;
    G4int fTransmissionGammaEntries;
    G4int fTransmissionPrimaryGammaEntries;
    G4int fSideScatterGammaEntries;
    G4int fSideScatterPrimaryGammaEntries;
};

}

#endif
