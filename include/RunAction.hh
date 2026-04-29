#ifndef B1RunAction_h
#define B1RunAction_h 1

#include "G4Accumulable.hh"
#include "G4UserRunAction.hh"
#include "globals.hh"

#include <string>

class G4Run;

namespace B1
{

class RunAction : public G4UserRunAction
{
  public:
    RunAction();
    ~RunAction() override;

    void BeginOfRunAction(const G4Run* run) override;
    void EndOfRunAction(const G4Run* run) override;

    void AddRunDetectorEdep(G4double edep);
    void AddRunDetectorGammaEntries(G4int n);
    void AddRunPrimaryGammaEntries(G4int n);

    void WriteEventData(G4int eventID,
                        G4double detectorEdep_keV,
                        G4int detectorGammaEntries,
                        G4int primaryGammaEntries,
                        G4int transmissionGammaEntries,
                        G4int transmissionPrimaryGammaEntries,
                        G4int sideScatterGammaEntries,
                        G4int sideScatterPrimaryGammaEntries);

    void WriteHitData(G4int eventID,
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
    void WriteMetadataFile(G4int nofEvents,
                           G4double totalEdep_keV,
                           G4double meanEdep_keV,
                           G4int totalGammaEntries,
                           G4int totalPrimaryEntries) const;

    std::string fEventCsvFileName;
    std::string fHitCsvFileName;
    std::string fMetadataFileName;

    G4Accumulable<G4double> fSumDetectorEdep;
    G4Accumulable<G4int> fSumDetectorGammaEntries;
    G4Accumulable<G4int> fSumPrimaryGammaEntries;
};

}

#endif
