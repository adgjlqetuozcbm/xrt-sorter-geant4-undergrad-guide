#ifndef PrimaryGeneratorAction_h
#define PrimaryGeneratorAction_h 1

#include "G4VUserPrimaryGeneratorAction.hh"
#include "ExperimentConfig.hh"
#include "G4ThreeVector.hh"
#include "globals.hh"

#include <vector>

class G4ParticleGun;
class G4Event;

struct SourcePhotonSample
{
  G4double energy_keV;
  G4double x_mm;
  G4double y_mm;
  G4double z_mm;
  G4double dir_x;
  G4double dir_y;
  G4double dir_z;
};

class PrimaryGeneratorAction : public G4VUserPrimaryGeneratorAction
{
  public:
    PrimaryGeneratorAction();
    ~PrimaryGeneratorAction() override;

    void GeneratePrimaries(G4Event* event) override;

    const G4ParticleGun* GetParticleGun() const { return fParticleGun; }

  private:
    void LoadSpectrumFile(const G4String& fileName);
    void LoadPhaseSpaceFile(const G4String& fileName);

    G4ParticleGun* fParticleGun;
    SourceMode fSourceMode;
    G4double fMonoEnergy_keV;

    // for mono and spectrum modes
    std::vector<G4double> fSpectrumEnergies_keV;
    std::vector<G4double> fSpectrumCDF;
    G4ThreeVector fSourcePos;
    G4ThreeVector fBeamDir;
    G4double fBeamHalfY_mm;
    G4double fBeamHalfZ_mm;

    // for phase_space mode
    std::vector<SourcePhotonSample> fSamples;
};

#endif
