using System;
using System.Collections.Generic;
using System.Linq;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Rooms;

namespace SpireHost.Driver.Combat.Tests;

/// <summary>
/// "callers": the OUT-OF-COMBAT game code that calls CardSelectCmd.From* and is driven by other teams' handlers
/// - the 22 events whose options open a card screen (grep of Models.Events/*.cs and EventModel.cs:510), the rest
/// site (Smith: FromDeckForUpgrade, cancelable) and the merchant (card removal: FromDeckForRemoval, manual
/// confirmation). Each is entered with the debug jump and played with a random policy for several policy
/// seeds, with this module's selector behind whatever room handler is registered (the temporary event / rest /
/// shop handlers in this console). The probe checks every CARD_SELECT frame that appears.
///
/// What it proves: prompts raised from blocking contexts inside event options, rest options and purchases are
/// captured (entry kind, prefs, cancelable ...) and answered without a fault in the selection code, including
/// the CANCEL exits that only the deck screens have.
/// What it does not judge: the events themselves. With the TEMPORARY event handler seven events abort inside
/// event code (UI singletons; fixed in the events team's own project): such an abort - a game-task fault whose
/// first frame is a method of Models.Events - is counted and reported, not failed. Any other abort fails.
/// </summary>
public static class CallerSuite
{
    /// <summary>Event model type names that call CardSelectCmd.From* (directly or through EventModel.SelectCardsToAddToDeckFromGrid).</summary>
    private static readonly string[] EventTypes =
    {
        "Amalgamator", "AromaOfChaos", "DoorsOfLightAndDark", "EndlessConveyor", "FieldOfManSizedHoles", "GraveOfTheForgotten", "LuminousChoir", "MorphicGrove",
        "SapphireSeed", "SelfHelpBook", "SpiralingWhirlpool", "SpiritGrafter", "StoneOfAllTime", "Symbiote", "Trial", "WaterloggedScriptorium", "Wellspring",
        "WhisperingHollow", "WoodCarvings", "ZenWeaver", "BrainLeech", "RoomFullOfCheese",
    };

    /// <summary>
    /// The presentation null-dereferences inside event code that the TEMPORARY event handler cannot get past
    /// (DRIVER-API.md section 12 "events": Amalgamator.CombineStrikes / CombineDefends, Trial.Accept /
    /// DoubleDown, WaterfallOfObviousSecrets.EnterWaterfall, DenseVegetation.Rest, JungleMazeAdventure.SafetyInNumbers,
    /// PunchOff.Nab, NCrystalSphereScreen.ShowScreen). Only a NullReferenceException whose first frame is one of
    /// these methods is tolerated.
    /// </summary>
    private static readonly string[] KnownEventSites =
    {
        "Models.Events.Amalgamator", "Models.Events.Trial.Accept", "Models.Events.Trial.DoubleDown", "Models.Events.WaterfallOfObviousSecrets.EnterWaterfall",
        "Models.Events.DenseVegetation.Rest", "Models.Events.JungleMazeAdventure.SafetyInNumbers", "Models.Events.PunchOff.Nab", "NCrystalSphereScreen.ShowScreen",
    };

    public static bool IsKnownEventSite(AbortInfo abort)
    {
        if (abort.Reason != AbortReason.GameTaskFault || abort.ExceptionType != "NullReferenceException" || abort.Frames.Count == 0)
        {
            return false;
        }
        foreach (string site in KnownEventSites)
        {
            if (abort.Frames[0].Contains(site, StringComparison.Ordinal))
            {
                return true;
            }
        }
        return false;
    }

    public static List<CaseResult> Run(HostSession session, string? only, int seeds, out object summary)
    {
        List<CaseResult> results = new List<CaseResult>();
        Dictionary<string, long> promptsByCaller = new Dictionary<string, long>();
        Dictionary<string, long> promptsByKind = new Dictionary<string, long>();
        Dictionary<string, long> endsByKind = new Dictionary<string, long>();
        Dictionary<string, long> foreignAborts = new Dictionary<string, long>();
        List<string> callersWithoutPrompt = new List<string>();
        List<string> missing = new List<string>();
        List<(string name, string character, Func<RunContext, System.Threading.Tasks.Task> setup)> cases = new List<(string, string, Func<RunContext, System.Threading.Tasks.Task>)>();
        foreach (string type in EventTypes)
        {
            EventModel? ev = ModelDb.AllEvents.FirstOrDefault(e => e.GetType().Name == type);
            if (ev == null)
            {
                missing.Add(type);
                continue;
            }
            string id = ev.Id.Entry;
            cases.Add(("event " + id, "IRONCLAD", async ctx =>
            {
                ScenarioTools.SetGold(ctx, 999);
                await ScenarioTools.JumpToEvent(ctx, id);
            }));
        }
        cases.Add(("rest site", "IRONCLAD", ctx => ScenarioTools.JumpToRoom(ctx, RoomType.RestSite)));
        cases.Add(("merchant", "SILENT", async ctx =>
        {
            ScenarioTools.SetGold(ctx, 999);
            await ScenarioTools.JumpToRoom(ctx, RoomType.Shop);
        }));

        ulong seed = 6000;
        foreach ((string name, string character, Func<RunContext, System.Threading.Tasks.Task> setup) in cases)
        {
            if (only != null && !name.Contains(only, StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }
            long prompts = 0;
            for (int k = 0; k < seeds; k++)
            {
                CombatProbe probe = new CombatProbe();
                PreferPolicy policy = new PreferPolicy(++seed, randomCombat: true);
                if (k % 2 == 0)
                {
                    // Every other seed steers towards the card screen instead of leaving it to chance: the first
                    // option of an event page (usually the one that opens it) and the merchant's card removal.
                    policy.Prefer = d =>
                    {
                        if (d.Kind == DecisionKind.Event)
                        {
                            return policy.FirstVerb(d, Verb.ChooseOption);
                        }
                        if (d.Kind == DecisionKind.Shop)
                        {
                            for (int i = 0; i < d.Actions.Count; i++)
                            {
                                if (d.Actions[i].Verb == Verb.ChooseOption && d.Actions[i].Source.Kind == RefKind.ShopEntry
                                    && d.Actions[i].Source.Object is MegaCrit.Sts2.Core.Entities.Merchant.MerchantCardRemovalEntry)
                                {
                                    return i;
                                }
                            }
                        }
                        return -1;
                    };
                }
                CaseResult r = CaseRunner.Run(session, "callers", name + " #" + k, character, setup, policy, probe, hp: 99999, maxTurns: 40, maxDecisions: 3000, seedLabel: "callers:" + name + ":" + k,
                    tolerateAbort: IsKnownEventSite);
                if (r.Passed && r.Outcome == "Aborted")
                {
                    string site = r.AbortFrames != null && r.AbortFrames.Count > 0 ? r.AbortFrames[0] : "?";
                    int paren = site.IndexOf('(');
                    site = paren > 0 ? site.Substring(0, paren) : site;
                    foreignAborts[site] = foreignAborts.GetValueOrDefault(site) + 1;
                }
                foreach (SelectRecord s in probe.Selects)
                {
                    prompts++;
                    string kind = s.Kind.ToString();
                    promptsByKind[kind] = promptsByKind.GetValueOrDefault(kind) + 1;
                    string end = kind + ":" + (s.End.Length > 0 ? s.End : "open");
                    endsByKind[end] = endsByKind.GetValueOrDefault(end) + 1;
                }
                results.Add(r);
            }
            promptsByCaller[name] = prompts;
            if (prompts == 0)
            {
                callersWithoutPrompt.Add(name);
            }
        }
        summary = new
        {
            callers = promptsByCaller.Count,
            seeds_per_caller = seeds,
            prompts_by_caller = promptsByCaller,
            prompts_by_kind = promptsByKind,
            prompt_ends = endsByKind,
            callers_without_prompt_in_these_seeds = callersWithoutPrompt,
            unknown_event_types = missing,
            aborts_inside_event_code_with_the_temporary_event_handler = foreignAborts,
        };
        return results;
    }
}
