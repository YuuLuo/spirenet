using System;
using System.Collections.Generic;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Runs.History;
using SpireHost.Driver;
using SpireHost.Obs.Schema;

namespace SpireHost.Obs.Encoding;

public sealed partial class ObsEncoder
{
    private readonly List<MapPoint> _nodes = new List<MapPoint>(96);
    private readonly HashSet<MapPoint> _reachable = new HashSet<MapPoint>(ReferenceEqualityComparer.Instance);
    private readonly List<long> _edges = new List<long>(192);

    // =====================================================================================================
    //  MAP_NODES / MAP_EDGES (schema 5.10): the current act's map only (NMapScreen shows nothing else)
    // =====================================================================================================

    private void EncodeMap(DecisionKind kind)
    {
        RunState run = _run;
        ActMap map = run.Map;
        _nodes.Clear();
        foreach (MapPoint p in map.GetAllMapPoints())
        {
            _nodes.Add(p);
        }
        AddUnique(map.StartingMapPoint);
        // regular nodes by (row, col); boss nodes appended last (boss, then second boss)
        _nodes.Remove(map.BossMapPoint);
        if (map.SecondBossMapPoint != null)
        {
            _nodes.Remove(map.SecondBossMapPoint);
        }
        _nodes.Sort(static (a, b) => a.coord.row != b.coord.row ? a.coord.row.CompareTo(b.coord.row) : a.coord.col.CompareTo(b.coord.col));
        _nodes.Add(map.BossMapPoint);
        if (map.SecondBossMapPoint != null)
        {
            _nodes.Add(map.SecondBossMapPoint);
        }

        Reachable(kind, run, map);
        List<MapPoint?>? votes = _d?.Context is MapContext mc && mc.SeatVotes.Count > 0 ? mc.SeatVotes : null;
        IReadOnlyList<MapCoord> visited = run.VisitedMapCoords;
        MapCoord? cur = run.CurrentMapCoord;
        IReadOnlyList<IReadOnlyList<MapPointHistoryEntry>> history = run.MapPointHistory;
        int act = run.CurrentActIndex;
        for (int i = 0; i < _nodes.Count; i++)
        {
            MapPoint p = _nodes[i];
            int r = _w.AddRow(T.MapNodes);
            _mapRow[p] = r;
            Span<int> row = _w.Row(T.MapNodes, r);
            row[MapNodeC.Row] = p.coord.row;
            row[MapNodeC.Col] = p.coord.col;
            row[MapNodeC.PointType] = Enum(Ev.MapPointType, (int)p.PointType);
            bool isCur = cur.HasValue && cur.Value == p.coord;
            bool wasVisited = Contains(visited, p.coord);
            row[MapNodeC.State] = isCur ? MapNodeState.Current : wasVisited ? MapNodeState.Visited : _reachable.Contains(p) ? MapNodeState.Reachable : MapNodeState.Normal;
            // a visited '?' shows the room it turned out to be: MapPointHistory[act][row].Rooms[0] (Nodes.Screens.Map/NNormalMapPoint.cs:327-335)
            if ((isCur || wasVisited) && p.PointType == MapPointType.Unknown && act < history.Count && p.coord.row >= 0 && p.coord.row < history[act].Count)
            {
                List<MapPointRoomHistoryEntry> rooms = history[act][p.coord.row].Rooms;
                if (rooms.Count > 0)
                {
                    row[MapNodeC.ResolvedRoom] = Enum(Ev.RoomType, (int)rooms[0].RoomType);
                }
            }
            int f = 0;
            if (p.Quests.Count > 0)
            {
                f |= MapNodeFlag.Quest;
            }
            if (ReferenceEquals(p, map.BossMapPoint))
            {
                f |= MapNodeFlag.Boss;
            }
            if (ReferenceEquals(p, map.SecondBossMapPoint))
            {
                f |= MapNodeFlag.SecondBoss;
            }
            if (ReferenceEquals(p, map.StartingMapPoint))
            {
                f |= MapNodeFlag.Start;
            }
            if (votes != null)
            {
                // Erratum H5 / schema 5.10: every player's map vote is shown on the map (NMapScreen.OnPlayerVoteChanged).
                for (int s = 0; s < votes.Count; s++)
                {
                    if (ReferenceEquals(votes[s], p))
                    {
                        f |= MapNodeFlag.VoteSeat0 << s;
                    }
                }
            }
            row[MapNodeC.Flags] = f;
        }
        _edges.Clear();
        for (int i = 0; i < _nodes.Count; i++)
        {
            foreach (MapPoint child in _nodes[i].Children)
            {
                if (_mapRow.TryGetValue(child, out int to))
                {
                    _edges.Add(((long)i << 32) | (uint)to);
                }
            }
        }
        _edges.Sort();
        for (int i = 0; i < _edges.Count; i++)
        {
            if (i > 0 && _edges[i] == _edges[i - 1])
            {
                continue;
            }
            int r = _w.AddRow(T.MapEdges);
            _w.Set(T.MapEdges, r, MapEdgeC.From, (int)(_edges[i] >> 32));
            _w.Set(T.MapEdges, r, MapEdgeC.To, (int)(_edges[i] & 0xFFFFFFFF));
        }
    }

    private void AddUnique(MapPoint? p)
    {
        if (p != null && !_nodes.Contains(p))
        {
            _nodes.Add(p);
        }
    }

    private static bool Contains(IReadOnlyList<MapCoord> list, MapCoord c)
    {
        for (int i = 0; i < list.Count; i++)
        {
            if (list[i] == c)
            {
                return true;
            }
        }
        return false;
    }

    /// <summary>
    /// state = 1 (reachable): on a MAP frame exactly what the real UI marks travelable (MapContext.Travelable, built by the
    /// driver's MapHandler from NMapScreen.RecalculateTravelability); on other frames the next step from the current node
    /// (children; nothing visited yet -> the starting node; last grid row -> boss; first of two bosses -> second boss).
    /// </summary>
    private void Reachable(DecisionKind kind, RunState run, ActMap map)
    {
        _reachable.Clear();
        if (kind == DecisionKind.Map && _d?.Context is MapContext mc)
        {
            foreach (MapPoint p in mc.Travelable)
            {
                _reachable.Add(p);
            }
            return;
        }
        IReadOnlyList<MapCoord> visited = run.VisitedMapCoords;
        if (visited.Count == 0)
        {
            if (map.StartingMapPoint != null)
            {
                _reachable.Add(map.StartingMapPoint);
            }
            return;
        }
        MapCoord last = visited[visited.Count - 1];
        if (map.SecondBossMapPoint != null && last == map.BossMapPoint.coord)
        {
            _reachable.Add(map.SecondBossMapPoint);
            return;
        }
        if (last.row == map.GetRowCount() - 1)
        {
            _reachable.Add(map.BossMapPoint);
            return;
        }
        MapPoint? point = map.GetPoint(last);
        if (point != null)
        {
            foreach (MapPoint c in point.Children)
            {
                _reachable.Add(c);
            }
        }
    }
}
