#pragma once

#include <madrona/components.hpp>
#include <madrona/math.hpp>
#include <madrona/physics.hpp>
#include <madrona/viz/system.hpp>

#include "consts.hpp"

namespace madEscape {

// Include several madrona types into the simulator namespace for convenience
using madrona::Entity;
using madrona::CountT;
using madrona::Query;
using madrona::base::Position;
using madrona::base::Rotation;
using madrona::base::Scale;
using madrona::base::ObjectID;
using madrona::phys::Velocity;
using madrona::phys::ResponseType;
using madrona::phys::ExternalForce;
using madrona::phys::ExternalTorque;


// WorldReset is a per-world singleton component that causes the current
// episode to be terminated and the world regenerated
// (Singleton components like WorldReset can be accessed via Context::singleton
// (eg ctx.singleton<WorldReset>().reset = 1)
struct WorldReset {
    int32_t reset;
};

// Discrete action component. Ranges are defined by consts::numMoveBuckets (5),
// repeated here for clarity
struct Action {
    int32_t moveAmount; // [0, 3]
    int32_t moveAngle; // [0, 7]
    int32_t rotate; // [-2, 2]
    int32_t grab; // 0 = do nothing, 1 = grab / release
};

// Per-agent reward
// Exported as an [N * A, 1] float tensor to training code
struct Reward {
    float v;
};

// Per-agent component that indicates that the agent's episode is finished
// This is exported per-agent for simplicity in the training code
struct Done {
    // Currently bool components are not supported due to
    // padding issues, so Done is an int32_t
    int32_t v;
};

// Observation state for the current agent.
// Positions are rescaled to the bounds of the play area to assist training.
struct SelfObservation {
    float roomX;
    float roomY;
    float globalX;
    float globalY;
    float globalZ;
    float maxY;
    float theta;
    float isGrabbing;
};

struct AgentID {
    int32_t id;
};

// The state of the world is passed to each agent in terms of egocentric
// polar coordinates. theta is degrees off agent forward.
struct PolarObservation {
    float r;
    float theta;
};

struct PartnerObservation {
    PolarObservation polar;
    float isGrabbing;
};

// Egocentric observations of other agents
struct PartnerObservations {
    PartnerObservation obs[consts::numAgents - 1];
};

// PartnerObservations is exported as a
// [N, A, consts::numAgents - 1, 3] // tensor to pytorch
static_assert(sizeof(PartnerObservations) == sizeof(float) *
    (consts::numAgents - 1) * 3);

// Per-agent egocentric observations for the interactable entities
// in the current room.
struct EntityObservation {
    PolarObservation polar;
    float encodedType;
};

struct RoomEntityObservations {
    EntityObservation obs[consts::maxObservationsPerAgent];
};

// RoomEntityObservations is exported as a
// [N, A, maxObservationsPerAgent, 3] tensor to pytorch
static_assert(sizeof(RoomEntityObservations) == sizeof(float) *
    consts::maxObservationsPerAgent * 3);

// Observation of the current room's door. It's relative position and
// whether or not it is ope
struct DoorObservation {
    PolarObservation polar;
    float isOpen; // 1.0 when open, 0.0 when closed.
};

struct LidarSample {
    float depth;
    float encodedType;
};

// Linear depth values and entity type in a circle around the agent
struct Lidar {
    LidarSample samples[consts::numLidarSamples];
};

// Number of steps remaining in the episode. Allows non-recurrent policies
// to track the progression of time.
struct StepsRemaining {
    uint32_t t;
};

// Tracks progress the agent has made through the challenge, used to add
// reward when more progress has been made
struct Progress {
    float maxY;
};

// Per-agent component storing Entity IDs of the other agents. Used to
// build the egocentric observations of their state.
struct OtherAgents {
    madrona::Entity e[consts::numAgents - 1];
};

// Tracks if an agent is currently grabbing another entity
struct GrabState {
    Entity constraintEntity;
};

// This enum is used to track the type of each entity for the purposes of
// classifying the objects hit by each lidar sample.
enum class EntityType : uint32_t {
    None,
    Button,
    Cube,
    Wall,
    Agent,
    Door,
    NumTypes,
};

// A per-door component that tracks whether or not the door should be open.
struct OpenState {
    bool isOpen;
};

// Linked buttons that control the door opening and whether or not the door
// should remain open after the buttons are pressed once.
struct DoorProperties {
    Entity buttons[consts::maxEntitiesPerRoom];
    int32_t numButtons;
    bool isPersistent;
};

// Similar to OpenState, true during frames where a button is pressed
struct ButtonState {
    bool isPressed;
};

// Room itself is not a component but is used by the singleton
// component "LevelState" (below) to represent the state of the full level
struct Room {
    // These are entities the agent will interact with
    Entity entities[consts::maxEntitiesPerRoom];

    // The walls that separate this room from the next
    Entity walls[2];

    // The door the agents need to figure out how to lower
    Entity door;
};

// A singleton component storing the state of all the rooms in the current
// randomly generated level
struct LevelState {
    Room rooms[consts::numRooms];
};

// Checkpoint structs.
struct ButtonSaveState {
    Position p;
    Rotation r;
    ButtonState b;
};

struct PhysicsEntityState {
    Position p;
    Rotation r;
    Velocity v;
};

struct DoorState {
    Position p;
    Rotation r;
    Velocity v;
    OpenState o;
};

struct AgentState {
    Position p;
    Rotation r;
    Velocity v;
    GrabState g;
    // These last aren't needed for restoring, but they are needed 
    // for determining if the state is worth restoring to.
    Reward re;
    Done d;
    StepsRemaining s;
    Progress pr;
};

struct Checkpoint {
    ButtonSaveState buttonStates[consts::numRooms * 2];
    PhysicsEntityState cubeStates[consts::numRooms * 3];
    DoorState doorStates[consts::numRooms];
    AgentState agentStates[consts::numAgents];
};

struct CheckpointReset {
    int32_t reset;
};

/* ECS Archetypes for the game */

// There are 2 Agents in the environment trying to get to the destination
struct Agent : public madrona::Archetype<
    // Basic components required for physics. Note that the current physics
    // implementation requires archetypes to have these components first
    // in this exact order.
    Position,
    Rotation,
    Scale,
    Velocity,
    ObjectID,
    ResponseType,
    madrona::phys::solver::SubstepPrevState,
    madrona::phys::solver::PreSolvePositional,
    madrona::phys::solver::PreSolveVelocity,
    ExternalForce,
    ExternalTorque,
    madrona::phys::broadphase::LeafID,

    // Internal logic state.
    GrabState,
    Progress,
    OtherAgents,
    EntityType,

    // Input
    Action,

    // Observations
    SelfObservation,
    AgentID,
    PartnerObservations,
    RoomEntityObservations,
    DoorObservation,
    Lidar,
    StepsRemaining,

    // Reward, episode termination
    Reward,
    Done,

    // Visualization: In addition to the fly camera, src/viewer.cpp can
    // view the scene from the perspective of entities with this component
    madrona::viz::VizCamera
> {};

// Archetype for the doors blocking the end of each challenge room
struct DoorEntity : public madrona::Archetype<
    Position, 
    Rotation,
    Scale,
    Velocity,
    ObjectID,
    ResponseType,
    madrona::phys::solver::SubstepPrevState,
    madrona::phys::solver::PreSolvePositional,
    madrona::phys::solver::PreSolveVelocity,
    ExternalForce,
    ExternalTorque,
    madrona::phys::broadphase::LeafID,
    OpenState,
    DoorProperties,
    EntityType
> {};

// Archetype for the button objects that open the doors
// Buttons don't have collision but are rendered
struct ButtonEntity : public madrona::Archetype<
    Position,
    Rotation,
    Scale,
    ObjectID,
    ButtonState,
    EntityType
> {};

// Generic archetype for entities that need physics but don't have custom
// logic associated with them.
struct PhysicsEntity : public madrona::Archetype<
    Position, 
    Rotation,
    Scale,
    Velocity,
    ObjectID,
    ResponseType,
    madrona::phys::solver::SubstepPrevState,
    madrona::phys::solver::PreSolvePositional,
    madrona::phys::solver::PreSolveVelocity,
    ExternalForce,
    ExternalTorque,
    madrona::phys::broadphase::LeafID,
    EntityType
> {};

// Tracks global progress the agent has made through the challenge, used to add
// reward when more progress has been made
struct GlobalProgress {
    float* progressPtr;
};

}
