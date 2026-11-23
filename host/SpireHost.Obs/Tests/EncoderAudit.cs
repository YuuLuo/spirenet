using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Reflection.Emit;
using System.Reflection.Metadata;
using System.Reflection.Metadata.Ecma335;
using System.Reflection.PortableExecutable;

namespace SpireHost.Obs.Tests;

/// <summary>
/// Static audit (obs-schema-v1 14.3): decodes the IL of every method of the encoder namespace (including compiler
/// generated closures nested in its types) and lists every referenced field / method whose declaring type or name is on
/// the hidden-information blacklist. Accesses that are allowed only in one place are allowed only there:
/// DrawPile -> EncodeSeats (count) and EncodePileZone (multiset), NextMove -> EncodeIntents (immediately .Intents).
///
/// Verifier finding R-7: since the encoder reads the decision contexts of the driver ("public views") instead of the
/// game, the reads that touch hidden data live in SpireHost.Driver, outside the scanned namespace, and the encoder
/// could reach three host-only members without any rule noticing. The audit therefore has three parts:
///   1. the encoder namespace against the blacklist, which now also names the host-only members of the driver
///      (CrystalContext.Minigame, ICardSelectHostView, Decision.Source);
///   2. the driver's public-view types (<see cref="PublicViewTypes"/>) against the same blacklist - the one place where a
///      hidden game member may be read is named: CrystalSphereCell.Item inside CrystalContext.FillPublicView, which reads
///      it only through a cell that is not hidden;
///   3. coverage: every type of SpireHost.Driver that the encoder references is either one of those public views or on
///      the short list of driver API types (<see cref="DriverApiTypes"/>), so that a new context cannot be read unaudited;
/// and a canary (<see cref="Canary.AuditCanary"/>) that reads forbidden members on purpose: the scan must report each of them.
/// </summary>
public static class EncoderAudit
{
    /// <summary>(type-or-member substring, allowed only inside these methods or null = never).</summary>
    private static readonly (string Pattern, string[]? AllowedIn)[] Rules =
    {
        ("MegaCrit.Sts2.Core.Random.Rng", null),
        ("RunRngSet", null),
        ("PlayerRngSet", null),
        ("::get_Rng", null),
        ("::get_Seed", null),
        ("::get_StringSeed", null),
        // the monster state machine namespace: only MoveState.Intents (the displayed intents) in EncodeIntents, see Check, and
        // the move forecast (ObsEncoder.Forecast.cs, RESET flag FORECAST, owner ruling 2026-11-17), which reads the machine;
        // the Rng rules above hold for it too
        ("MonsterMoves.MonsterMoveStateMachine.", new[] { "Forecast" }),
        ("::get_StateLog", new[] { "EncodeForecast" }),
        ("::get_FollowUpState", new[] { "ForecastNext" }),
        ("RoomSet", null),
        ("::_rooms", null),
        ("RelicGrabBag", null),
        ("PlayerOdds", null),
        ("::_internalData", null),
        ("CombatState::get_Encounter", null),
        ("::ToSave", null),
        ("::ToSerializable", null),
        ("NetFullCombatState", null),
        ("RunSignature", null),
        ("StateSignature", null),
        ("::get_CombatId", null),
        // crystal sphere: the item layout is hidden (CrystalSphereItem.PlaceItem); only IsHidden / DivinationCount / the tool
        ("CrystalSphereCell::get_Item", null),
        ("CrystalSphereMinigame::get_Items", null),
        ("CrystalSphereMinigame::_items", null),
        ("CrystalSphereMinigame::get_Rng", null),
        ("::get_NextMove", new[] { "EncodeIntents", "EncodeForecast" }),
        ("::get_DrawPile", new[] { "EncodeSeats", "EncodePileZone" }),
        // host-only members of the driver's decision model (R-7): the live minigame behind the crystal view, the raw (game
        // order) candidates of a selection frame, and the frame / handler object itself
        ("SpireHost.Driver.CrystalContext::Minigame", null),
        ("SpireHost.Driver.ICardSelectHostView::", null),
        ("SpireHost.Driver.Decision::Source", null),
    };

    /// <summary>
    /// The same blacklist for the driver's public views. Differences: a view may read the one hidden member its own
    /// filter guards (CrystalSphereCell.Item in FillPublicView, behind !IsHidden), and CrystalContext.FillPublicView is
    /// the method that reads CrystalContext.Minigame by design.
    /// </summary>
    private static readonly (string Pattern, string[]? AllowedIn)[] ViewRules = BuildViewRules();

    private static (string, string[]?)[] BuildViewRules()
    {
        List<(string, string[]?)> rules = new List<(string, string[]?)>();
        foreach ((string pattern, string[]? allowed) in Rules)
        {
            if (pattern == "CrystalSphereCell::get_Item")
            {
                rules.Add((pattern, new[] { "CrystalContext::FillPublicView" }));
            }
            else if (pattern == "SpireHost.Driver.CrystalContext::Minigame")
            {
                rules.Add((pattern, new[] { "CrystalContext::FillPublicView" }));
            }
            else if (pattern == "::get_NextMove" || pattern == "::get_DrawPile")
            {
                rules.Add((pattern, null));   // the encoder's two allowances are the encoder's, not a view's
            }
            else
            {
                rules.Add((pattern, allowed));
            }
        }
        return rules.ToArray();
    }

    /// <summary>
    /// The decision contexts and view helpers of SpireHost.Driver that an encoder reads (DecisionContexts.cs, LocText.cs), and
    /// HookCache (HookCache*.cs: the encoder's and the COMBAT handler's transcriptions of game hooks over cached listener
    /// lists; it moved from the encoder namespace to the driver so that both can use it, and is audited here with the view
    /// rules, which are stricter than the encoder's: no DrawPile / NextMove read at all).
    /// </summary>
    public static readonly string[] PublicViewTypes =
    {
        "MapContext", "CombatSeatInfo", "CombatContext", "CardSelectContext", "RewardContext", "CardRewardContext", "ShopContext", "RestContext", "TreasureContext",
        "ConfirmContext", "ConfirmKeys", "BundleContext", "EventOptionInfo", "EventContext", "CrystalCellView", "CrystalContext", "LocText", "HookCache",
    };

    /// <summary>
    /// Driver API types the encoder may reference besides the public views: the decision model (kinds, verbs, references,
    /// the action list), the run context and episode bookkeeping it is handed, and the registration / counter contracts.
    /// None of them is a view of game state; RunContext hands out the live run, which part 1 audits member by member.
    /// </summary>
    public static readonly string[] DriverApiTypes =
    {
        "Decision", "DecisionKind", "LegalAction", "GameRef", "RefKind", "Verb", "ActionFlags", "RunContext", "HostSession", "EpisodeResult", "EpisodeOutcome", "EpisodeOptions", "EpisodeSpec",
        "EpisodeCounters", "AbortInfo", "AbortReason", "IPolicy", "IEpisodeObserver", "IDriverModule", "IModuleCounters", "DriverRegistry", "CanonicalCardClasses", "CardSelectEntry",
        "ChoiceContextKind", "NotReady", "PolicyStopException", "EpisodeAbortException", "StrictCounters", "IDecisionSource", "IPendingDecision", "RoomScreen",
    };

    private static readonly Dictionary<short, OperandType> Operands = BuildOperands();

    private static Dictionary<short, OperandType> BuildOperands()
    {
        Dictionary<short, OperandType> d = new Dictionary<short, OperandType>();
        foreach (FieldInfo f in typeof(OpCodes).GetFields(BindingFlags.Public | BindingFlags.Static))
        {
            if (f.GetValue(null) is OpCode op)
            {
                d[op.Value] = op.OperandType;
            }
        }
        return d;
    }

    public static List<string> Scan(Assembly asm, string ns, out int methods, out int refs) => Scan(asm, ns, null, Rules, out methods, out refs, null);

    /// <summary>Part 2: the driver's public-view types (by simple name, closures included) against the view rules.</summary>
    public static List<string> ScanPublicViews(Assembly driver, out int methods, out int refs) => Scan(driver, "SpireHost.Driver", new HashSet<string>(PublicViewTypes, StringComparer.Ordinal), ViewRules, out methods, out refs, null);

    /// <summary>Part 3: types of SpireHost.Driver that the encoder namespace references and that are neither a public view nor a driver API type.</summary>
    public static List<string> UnauditedDriverTypes(Assembly encoder, string ns)
    {
        HashSet<string> referenced = new HashSet<string>(StringComparer.Ordinal);
        Scan(encoder, ns, null, Array.Empty<(string, string[]?)>(), out _, out _, referenced);
        HashSet<string> known = new HashSet<string>(PublicViewTypes, StringComparer.Ordinal);
        known.UnionWith(DriverApiTypes);
        List<string> unknown = new List<string>();
        foreach (string member in referenced)
        {
            // "SpireHost.Driver.TypeName::member" (the driver's own namespace only, not its sub-namespaces)
            const string prefix = "SpireHost.Driver.";
            if (!member.StartsWith(prefix, StringComparison.Ordinal))
            {
                continue;
            }
            int sep = member.IndexOf("::", StringComparison.Ordinal);
            string type = member.Substring(prefix.Length, (sep < 0 ? member.Length : sep) - prefix.Length);
            if (type.Contains('.') || known.Contains(type))
            {
                continue;
            }
            if (!unknown.Contains(type))
            {
                unknown.Add(type);
            }
        }
        unknown.Sort(StringComparer.Ordinal);
        return unknown;
    }

    private static List<string> Scan(Assembly asm, string ns, HashSet<string>? onlyTypes, (string Pattern, string[]? AllowedIn)[] rules, out int methods, out int refs, HashSet<string>? collect)
    {
        List<string> hits = new List<string>();
        methods = 0;
        refs = 0;
        using FileStream fs = File.OpenRead(asm.Location);
        using PEReader pe = new PEReader(fs);
        MetadataReader md = pe.GetMetadataReader();
        foreach (TypeDefinitionHandle th in md.TypeDefinitions)
        {
            TypeDefinition td = md.GetTypeDefinition(th);
            if (!InNamespace(md, td, ns))
            {
                continue;
            }
            if (onlyTypes != null && !onlyTypes.Contains(OutermostName(md, td)))
            {
                continue;
            }
            foreach (MethodDefinitionHandle mh in td.GetMethods())
            {
                MethodDefinition m = md.GetMethodDefinition(mh);
                if (m.RelativeVirtualAddress == 0)
                {
                    continue;
                }
                methods++;
                string owner = OwnerMethodName(md, td, m);
                MethodBodyBlock body = pe.GetMethodBody(m.RelativeVirtualAddress);
                BlobReader il = body.GetILReader();
                while (il.RemainingBytes > 0)
                {
                    short code = il.ReadByte();
                    if (code == 0xFE)
                    {
                        code = unchecked((short)(0xFE00 | il.ReadByte()));
                    }
                    OperandType ot = Operands.TryGetValue(code, out OperandType o) ? o : OperandType.InlineNone;
                    switch (ot)
                    {
                        case OperandType.InlineField:
                        case OperandType.InlineMethod:
                        case OperandType.InlineTok:
                        {
                            int token = il.ReadInt32();
                            string? name = MemberName(md, MetadataTokens.EntityHandle(token));
                            if (name != null)
                            {
                                refs++;
                                collect?.Add(name);
                                Check(name, owner, hits, rules);
                            }
                            break;
                        }
                        case OperandType.InlineNone:
                            break;
                        case OperandType.ShortInlineBrTarget:
                        case OperandType.ShortInlineI:
                        case OperandType.ShortInlineVar:
                            il.ReadByte();
                            break;
                        case OperandType.InlineVar:
                            il.ReadInt16();
                            break;
                        case OperandType.InlineI8:
                        case OperandType.InlineR:
                            il.ReadInt64();
                            break;
                        case OperandType.InlineSwitch:
                            int n = il.ReadInt32();
                            for (int i = 0; i < n; i++)
                            {
                                il.ReadInt32();
                            }
                            break;
                        default:
                            il.ReadInt32();
                            break;
                    }
                }
            }
        }
        return hits;
    }

    private static void Check(string member, string owner, List<string> hits, (string Pattern, string[]? AllowedIn)[] rules)
    {
        foreach ((string pattern, string[]? allowed) in rules)
        {
            if (member.Contains(pattern, StringComparison.Ordinal))
            {
                if (pattern == "MonsterMoves.MonsterMoveStateMachine." && member.EndsWith(".MoveState::get_Intents", StringComparison.Ordinal) && owner.Contains("EncodeIntents", StringComparison.Ordinal))
                {
                    continue; // NextMove.Intents: exactly what NCreature shows (Nodes.Combat/NCreature.cs:458)
                }
                if (allowed != null && Array.Exists(allowed, a => owner.Contains(a, StringComparison.Ordinal)))
                {
                    continue;
                }
                hits.Add(owner + " -> " + member);
            }
        }
    }

    /// <summary>The simple name of the outermost declaring type (a closure class nested in a view belongs to that view).</summary>
    private static string OutermostName(MetadataReader md, TypeDefinition td)
    {
        while (true)
        {
            TypeDefinitionHandle decl = td.GetDeclaringType();
            if (decl.IsNil)
            {
                return md.GetString(td.Name);
            }
            td = md.GetTypeDefinition(decl);
        }
    }

    private static bool InNamespace(MetadataReader md, TypeDefinition td, string ns)
    {
        while (true)
        {
            if (md.GetString(td.Namespace) == ns)
            {
                return true;
            }
            TypeDefinitionHandle decl = td.GetDeclaringType();
            if (decl.IsNil)
            {
                return false;
            }
            td = md.GetTypeDefinition(decl);
        }
    }

    private static string OwnerMethodName(MetadataReader md, TypeDefinition td, MethodDefinition m)
    {
        // closures: <EncodeIntents>b__12_0 inside a nested display class -> the method name carries the user method
        return md.GetString(td.Name) + "::" + md.GetString(m.Name);
    }

    /// <summary>The canary: its namespace is scanned with both rule sets; every forbidden member it reads must be reported.</summary>
    public static List<string> ScanCanary(bool viewRules, out int methods)
    {
        return Scan(typeof(Canary.AuditCanary).Assembly, "SpireHost.Obs.Tests.Canary", null, viewRules ? ViewRules : Rules, out methods, out _, null);
    }

    private static string? MemberName(MetadataReader md, EntityHandle h)
    {
        switch (h.Kind)
        {
            case HandleKind.MemberReference:
            {
                MemberReference mr = md.GetMemberReference((MemberReferenceHandle)h);
                return TypeName(md, mr.Parent) + "::" + md.GetString(mr.Name);
            }
            case HandleKind.MethodDefinition:
            {
                MethodDefinition d = md.GetMethodDefinition((MethodDefinitionHandle)h);
                return TypeName(md, d.GetDeclaringType()) + "::" + md.GetString(d.Name);
            }
            case HandleKind.FieldDefinition:
            {
                FieldDefinition f = md.GetFieldDefinition((FieldDefinitionHandle)h);
                return TypeName(md, f.GetDeclaringType()) + "::" + md.GetString(f.Name);
            }
            case HandleKind.MethodSpecification:
            {
                MethodSpecification s = md.GetMethodSpecification((MethodSpecificationHandle)h);
                return MemberName(md, s.Method);
            }
            default:
                return null;
        }
    }

    private static string TypeName(MetadataReader md, EntityHandle h)
    {
        switch (h.Kind)
        {
            case HandleKind.TypeReference:
            {
                TypeReference tr = md.GetTypeReference((TypeReferenceHandle)h);
                return md.GetString(tr.Namespace) + "." + md.GetString(tr.Name);
            }
            case HandleKind.TypeDefinition:
            {
                TypeDefinition td = md.GetTypeDefinition((TypeDefinitionHandle)h);
                return md.GetString(td.Namespace) + "." + md.GetString(td.Name);
            }
            case HandleKind.TypeSpecification:
            {
                // generic instantiation: decode the generic type definition's name from the signature blob
                TypeSpecification ts = md.GetTypeSpecification((TypeSpecificationHandle)h);
                BlobReader br = md.GetBlobReader(ts.Signature);
                SignatureTypeCode c = br.ReadSignatureTypeCode();
                if (c == SignatureTypeCode.GenericTypeInstance)
                {
                    br.ReadSignatureTypeCode();
                    return TypeName(md, br.ReadTypeHandle());
                }
                return "typespec";
            }
            default:
                return "?";
        }
    }
}
